use anyhow;
use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{EventLoop, StreamData, UnknownTypeInputBuffer};
use glutin_window::GlutinWindow as Window;
use graphics;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rustfft::num_complex::Complex32;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
use rustfft::FFT;

use std::f32::consts as f32_consts;
use std::fmt::Debug;
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::thread;

const FPS: u64 = 15;
const UPS: u64 = 15;
const WINDOW_SIZE: (usize, usize) = (640, 480);

const STORE_SAMPLES: usize = 32768; // at 44khz, this is ~1300ms

// for a real input signal (imaginary parts all zero) the second half of the FFT
// (bins from N / 2 + 1 to N - 1) contain no useful additional information (they
// have complex conjugate symmetry with the first N / 2 - 1 bins). The last
// useful bin (for practical aplications) is at N / 2 - 1
const FFT_SIZE: usize = STORE_SAMPLES / 2 - 1;

// how often to rerun the fft
const REACT_SAMPLES: u32 = 512; // at 44khz, this is ~86ms

#[derive(Debug)]
struct FftOutput {
    samples: Vec<f32>,
    fft: Vec<f32>,
}

impl FftOutput {
    fn new() -> Self {
        Self {
            samples: vec![0.0; STORE_SAMPLES],
            fft: vec![0.0; FFT_SIZE],
        }
    }
}

struct AudioThread {
    receiver: Arc<RwLock<FftOutput>>,
}

impl AudioThread {
    pub fn new() -> Result<Self, anyhow::Error> {
        let receiver = Arc::new(RwLock::new(FftOutput::new()));
        let c_receiver = receiver.clone();

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no default audio device");
        let event_loop = host.event_loop();
        let format = device.default_input_format()?;
        println!("format: {:?}", format);
        let _stream_id = event_loop.build_input_stream(&device, &format);

        let mut planner: FFTplanner<f32> = FFTplanner::new(false);
        let fft_planner = planner.plan_fft(STORE_SAMPLES);

        let fft_sender = Self::fft_thread(fft_planner, c_receiver);

        thread::Builder::new()
            .name("audio thread".into())
            .spawn(move || {
                Self::audio_thread(event_loop, fft_sender);
            })
            .expect("failed spawn audio thread");

        Ok(Self { receiver })
    }

    fn audio_thread(
        event_loop: EventLoop,
        fft_sender: mpsc::SyncSender<Vec<f32>>,
    ) {
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
                    .map(|b| {
                        rescale(*b, (u16::MIN, u16::MAX), (-1.0f32, 1.0f32))
                    })
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
                    let it: Vec<f32> = samples[idx..]
                        .iter()
                        .chain(samples[..idx].iter())
                        .copied()
                        .collect();
                    fft_sender.send(it).expect("fft thread hung up");
                }
            }
        });
    }

    fn fft_thread(
        fft_planner: Arc<dyn FFT<f32>>,
        receiver: Arc<RwLock<FftOutput>>,
    ) -> mpsc::SyncSender<Vec<f32>> {
        // we don't want this to be unbounded. we want the audio thread to block
        // (and thus miss samples) when we get behind. but we do want them to be
        // able to run a bit in parallel
        let (tx, rx) = mpsc::sync_channel(2);

        thread::Builder::new()
            .name("fft thread".into())
            .spawn(move || {
                for samples in rx {
                    let fft = Self::compute_fft(samples, &fft_planner);
                    Self::send_fft(fft, &receiver);
                }
            })
            .expect("failed spawn fft thread");
        tx
    }

    fn compute_fft<'a>(
        samples: Vec<f32>,
        fft_planner: &'a Arc<dyn FFT<f32>>,
    ) -> FftOutput {
        let hammed = samples.iter().enumerate().map(|(i, dp)| {
            // https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
            const A0: f32 = 0.53836;
            const TAU: f32 = f32_consts::PI * 2.0;
            *dp * (A0
                - ((1.0 - A0)
                    * f32::cos(TAU * i as f32 / samples.len() as f32)))
        });

        let as_complex = hammed.map(|c| Complex32::new(c, 0.0));
        let mut fft_samples: Vec<Complex32> = as_complex.collect();
        let mut fft_output: Vec<Complex32> =
            vec![Zero::zero(); fft_samples.len()];
        fft_planner.process(&mut fft_samples, &mut fft_output);

        fft_output.truncate(FFT_SIZE);

        let fft = fft_output
            .iter()
            .map(|elem| {
                // normalise it
                let amplitude = elem.norm_sqr().sqrt();
                let normal = amplitude / (samples.len() as f32).sqrt();
                normal
            })
            .collect();

        FftOutput { samples, fft }
    }

    fn send_fft(output: FftOutput, receiver: &Arc<RwLock<FftOutput>>) {
        let mut lock = receiver.write().expect("failed writer lock");
        *lock = output;
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
        // we want to let go of that lock as fast as possible, so compute the
        // lines we need to draw and then release it while we draw them
        let vp = args.viewport();
        let (height, width) = (vp.window_size[1], vp.window_size[0]);
        let lines: Vec<([f32; 4], [f64; 4])> = {
            let data = self.audio_thread.receiver.read().expect("read lock");

            let fft_lines = data.fft[1..].iter().enumerate().map(|(i, dp)| {
                // TODO resize fft output so we can draw fewer of these
                // lines. we're trying to draw 16k lines in the 1k-ish
                // pixels we have
                let col = rescale(
                    i as f32,
                    (0f32, data.fft.len() as f32 - 1f32),
                    (0f32, width as f32),
                ) as f64;
                let depth =
                    rescale(*dp, (0.0, 1.0), (0.0, height as f32)) as f64;
                let colour = [0.0, 0.0, 1.0, 1.0];
                let line = [col, 0.0, col, depth];
                (colour, line)
            });

            let sample_lines =
                data.samples.iter().enumerate().map(|(i, dp)| {
                    let col = rescale(
                        i as f32,
                        (0.0, data.samples.len() as f32 - 1.0),
                        (0.0, width as f32),
                    ) as f64;
                    let row = rescale(
                        *dp,
                        (-1.0, 1.0),
                        (0.0, height as f32),
                    ) as f64;
                    let colour = [1.0, 0.0, 0.0, 1.0];
                    let line = [col, row, col, height as f64/2.0];
                    (colour, line)
                });

            fft_lines.chain(sample_lines).collect()
        };

        self.gl.draw(vp, |c, gl| {
            const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
            graphics::clear(BLACK, gl);
            for (colour, line) in lines {
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
        WindowSettings::new("noise", [row_px as u32, col_px as u32])
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
    F: Into<f32> + Copy + PartialOrd + Debug,
    T: Into<f32> + From<f32> + Copy + Debug,
{
    let (from_min, from_max) = from_range;
    let from_span: f32 = from_max.into() - from_min.into();

    let percent: f32 = (num.into() - from_min.into()) / from_span;

    let (to_min, to_max) = to_range;
    let to_span: f32 = to_max.into() - to_min.into();

    let output = (percent * to_span + to_min.into()).into();

    if cfg!(debug_assertions) && (num < from_min || num > from_max) {
        eprintln!(
            "rescale not happy about {:?} < {:?} < {:?} --> {:?} < {:?} < {:?}",
            from_min, num, from_max, to_min, output, to_max,
        );
    }

    output
}

#[test]
fn test_rescale() {
    assert_eq!(rescale(5u8, (0u8, 10u8), (-1f32, 1f32)), 0.0);
    assert_eq!(rescale(0.5, (-1f32, 1f32), (0f32, 1f32)), 0.75);
    assert_eq!(rescale(0.5, (-1f32, 1f32), (0f32, 200f32)), 150f32);
    assert_eq!(rescale(0.25, (-1f32, 1f32), (1f32, -1f32)), -0.25);
}
