use anyhow;
use cpal;
use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{EventLoop, StreamData, UnknownTypeInputBuffer};
use glutin_window::GlutinWindow as Window;
use graphics::types::Rectangle;
use graphics::{self, Transformed};
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rustfft::num_complex::Complex32;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
use rustfft::FFT;

use std::f32::consts as f32_consts;
use std::sync::{Arc, RwLock};
use std::thread;

const FPS: u64 = 15;
const UPS: u64 = 15;
const WINDOW_SIZE: (usize, usize) = (200, 200);
const PIXEL_SIZE: u32 = 2;

const STORE_SAMPLES: usize = 16384; // at 44khz, this is ~372ms
const REACT_SAMPLES: usize = 512; // at 44khz, this is ~11ms

type FftOutput = Vec<f32>;

struct AudioThread {
    receiver: Arc<RwLock<FftOutput>>,
}

impl AudioThread {
    pub fn new() -> Result<Self, anyhow::Error> {
        let receiver = Arc::new(RwLock::new(vec![0.0; STORE_SAMPLES]));
        let c_receiver = receiver.clone();

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no default audio device");
        let event_loop = host.event_loop();
        let format = device.default_input_format()?;
        println!("format: {:?}", format);
        let _stream_id = event_loop.build_input_stream(&device, &format);

        thread::spawn(move || {
            Self::audio_thread(event_loop, c_receiver);
        });

        Ok(Self { receiver })
    }

    fn audio_thread(event_loop: EventLoop, receiver: Arc<RwLock<FftOutput>>) {
        let mut planner: FFTplanner<f32> = FFTplanner::new(false);
        let fft = planner.plan_fft(STORE_SAMPLES);
        let mut samples: Vec<f32> = vec![0f32; STORE_SAMPLES];
        let mut idx = 0;
        let mut react = 0;

        event_loop.run(move |stream_id, stream_result| {
            // react to stream events and read or write stream data here
            let stream_data = match stream_result {
                Ok(data) => data,
                Err(err) => {
                    eprintln!(
                        "an error occurred on stream {:?}: {}",
                        stream_id, err
                    );
                    return;
                }
            };

            // https://docs.rs/cpal/0.8.2/cpal/enum.SampleFormat.html
            let new_samples: Vec<f32> = match stream_data {
                StreamData::Input {
                    buffer: UnknownTypeInputBuffer::U16(buf),
                } => buf
                    .iter()
                    .map(|b| rescale(*b, (0u16, 65535u16), (-1.0f32, 1.0f32)))
                    .collect(),
                StreamData::Input {
                    buffer: UnknownTypeInputBuffer::I16(buf),
                } => buf
                    .iter()
                    .map(|b| {
                        rescale(*b, (i16::MIN, i16::MAX), (-1.0f32, 1.0f32))
                    })
                    .collect(),
                StreamData::Input {
                    buffer: UnknownTypeInputBuffer::F32(buf),
                } => buf.iter().map(|b| *b as f32).collect(),
                StreamData::Output { .. } => unreachable!("got output data?"),
            };

            for b in new_samples {
                samples[idx] = b;
                idx += 1;
                idx %= STORE_SAMPLES;

                react += 1;
                react %= REACT_SAMPLES;

                if react == 0 {
                    // it's time to send an fft! the pointer is probably in the
                    // middle of the sample buffer but that represents a
                    // discontinuity so we actually need to chain the
                    // before/after bits
                    let it = samples[idx..].iter().chain(samples[..idx].iter());
                    Self::send_fft(it.copied(), &fft, &receiver);
                }

                /*samples.push(b);

                if samples.len() >= NUM_SAMPLES {
                    Self::send_fft(samples.iter().copied(), &fft, &receiver);

                    samples.truncate(0); // leaves the allocation though
                }*/
            }
        });
    }

    fn send_fft(
        samples: impl Iterator<Item = f32>,
        fft: &Arc<dyn FFT<f32>>,
        receiver: &Arc<RwLock<FftOutput>>,
    ) {
        let mut fft_output: Vec<Complex32> = vec![Zero::zero(); STORE_SAMPLES];

        let hammed = samples.enumerate().map(|(i, dp)| {
            // https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
            const A0: f32 = 0.53836;
            const TAU: f32 = f32_consts::PI * 2.0;
            dp * (A0
                - ((1.0 - A0) * f32::cos(TAU * i as f32 / STORE_SAMPLES as f32)))
        });

        let as_complex = hammed.map(|c| Complex32::new(c, 0.0));
        let mut fft_samples: Vec<Complex32> = as_complex.collect();
        fft.process(&mut fft_samples, &mut fft_output);

        let normalised = fft_output.iter().map(|cm| cm.re).collect::<Vec<_>>();

        let mut lock = receiver.write().expect("failed writer lock");
        lock[..].clone_from_slice(&normalised);
        drop(lock);
    }
}

pub struct App {
    gl: GlGraphics,
    audio_thread: AudioThread,
}

impl App {
    fn new(gl: GlGraphics, audio_thread: AudioThread) -> Self {
        Self { gl, audio_thread }
    }

    fn render(&mut self, args: RenderArgs) {
        let data = self.audio_thread.receiver.read().expect("read lock");

        let vp = args.viewport();
        self.gl.draw(vp, |c, gl| {
            const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
            graphics::clear(BLACK, gl);

            let (rows, cols) = WINDOW_SIZE;

            for (i, dp) in data.iter().enumerate() {
                let col = (i as f64 / data.len() as f64) * cols as f64;
                let depth = *dp as f64 * rows as f64;

                //let depth =
                //    rescale(*dp, (-1f32, 1f32), (0.0f32, rows as f32)) as f64;
                let blueness = rescale(*dp, (-1f32, 1f32), (0f32, 1f32));

                let colour = [0.0, 0.0, blueness, 1.0];
                let line = [0.0, col, depth, col];
                graphics::line(colour, 1.0, line, c.transform, gl)
            }
        });
    }

    fn update(&mut self, _args: UpdateArgs) {}
}

fn main() -> Result<(), anyhow::Error> {
    let (row_px, col_px) = WINDOW_SIZE;

    let audio_thread = AudioThread::new()?;

    let opengl = OpenGL::V3_2;
    let mut window: Window =
        WindowSettings::new("noise", [col_px as u32, row_px as u32])
            .graphics_api(opengl)
            .exit_on_esc(true)
            .build()
            .expect("failed to build window");

    let mut app = App::new(GlGraphics::new(opengl), audio_thread);

    let mut settings = EventSettings::new();
    settings.max_fps = FPS;
    settings.ups = UPS;
    let mut events = Events::new(settings);
    // handle UI events on the main thread
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(args);
        }
        if let Some(args) = e.update_args() {
            app.update(args);
        }
    }

    Ok(())
}

fn rescale<F, T>(num: F, from_range: (F, F), to_range: (T, T)) -> T
where
    F: Into<f32> + Copy,
    T: Into<f32> + From<f32> + Copy,
{
    // 0, 10
    let (from_min, from_max) = from_range;
    // 10
    let from_span: f32 = from_max.into() - from_min.into();

    // (5 - 0) / 10 == 0.5
    let percent: f32 = (num.into() - from_min.into()) / from_span;

    // -1, 1
    let (to_min, to_max) = to_range;
    // 2.0
    let to_span: f32 = to_max.into() - to_min.into();

    // 0.5 * 2 + -1 == 0
    (percent * to_span + to_min.into()).into()
}

#[test]
fn test_rescale() {
    assert_eq!(rescale(5u8, (0u8, 10u8), (-1f32, 1f32)), 0.0f32);
    assert_eq!(rescale(0.5, (-1f32, 1f32), (0f32, 1f32)), 0.75f32);
    assert_eq!(rescale(0.5, (-1f32, 1f32), (0f32, 200f32)), 150);
}
