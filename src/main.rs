use anyhow;
use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{EventLoop, StreamData, UnknownTypeInputBuffer};
use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan32};
use fftw::types::{c32, Flag, Sign};
use graphics;
use num_traits::Float;
use glium_graphics::{GliumWindow, OpenGL, Glium2d};
use piston::input::{RenderArgs, RenderEvent};
use piston::window::WindowSettings;

use std::env::consts::OS;
use std::f32::consts::PI;
use std::f64::consts::PI as PI64;
use std::fmt::Debug;
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::thread;

const FPS: u64 = 15;
const WINDOW_SIZE: (usize, usize) = (640, 480);

const STORE_SAMPLES: usize = 65536; // at 44khz, this is ~743ms
const FFT_SAMPLES: usize = 16384;

// for a real input signal (imaginary parts all zero) the second half of the FFT
// (bins from N / 2 + 1 to N - 1) contain no useful additional information (they
// have complex conjugate symmetry with the first N / 2 - 1 bins). The last
// useful bin (for practical aplications) is at N / 2 - 1
const FFT_SIZE: usize = FFT_SAMPLES / 2 - 1;

// how often to rerun the fft
const REACT_SAMPLES: usize = 512; // at 44khz, this is ~12ms

// how many samples we consider to determine the current volume
const RECENT_VOLUME_SAMPLES: usize = 2048;
const MAX_VOLUME_SAMPLES: usize = STORE_SAMPLES;

type Line = [f64; 4];
type Colour = [f32; 4];

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

        let fft_sender = Self::fft_thread(c_receiver);

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
        let mut samples: Vec<f32> = vec![0.0; STORE_SAMPLES];
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
                    .map(|b| rescale(*b, (u16::MIN, u16::MAX), (-1.0, 1.0)))
                    .collect(),
                StreamData::Input {
                    buffer: UnknownTypeInputBuffer::I16(buf),
                } => buf
                    .iter()
                    .map(|b| rescale(*b, (i16::MIN, i16::MAX), (-1.0, 1.0)))
                    .collect(),
                StreamData::Input {
                    buffer: UnknownTypeInputBuffer::F32(buf),
                } => buf.to_vec(),
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
        receiver: Arc<RwLock<FftOutput>>,
    ) -> mpsc::SyncSender<Vec<f32>> {
        // we don't want this to be unbounded because we want the audio thread
        // to block (and thus miss samples) when we get behind. but we do want
        // them to be able to run a bit in parallel
        let (tx, rx) = mpsc::sync_channel(2);

        thread::Builder::new()
            .name("fft thread".into())
            .spawn(move || {
                let mut input = AlignedVec::new(FFT_SAMPLES);
                let mut output = AlignedVec::new(FFT_SAMPLES);
                let mut plan: C2CPlan32 = C2CPlan::aligned(
                    &[FFT_SAMPLES],
                    Sign::Forward,
                    Flag::MEASURE | Flag::DESTROYINPUT,
                )
                .expect("couldn't plan fft");

                for samples in rx {
                    let fft = Self::compute_fft(
                        &mut input,
                        &mut output,
                        samples,
                        &mut plan,
                    );
                    Self::send_fft(fft, &receiver);
                }
            })
            .expect("failed spawn fft thread");
        tx
    }

    fn compute_fft<'a>(
        mut input_buffer: &mut AlignedVec<c32>,
        mut output_buffer: &mut AlignedVec<c32>,
        samples: Vec<f32>,
        plan: &mut C2CPlan32,
    ) -> FftOutput {
        for (i, dp) in samples[STORE_SAMPLES - FFT_SAMPLES..].iter().enumerate()
        {
            // https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
            const A0: f32 = 0.53836;
            const TAU: f32 = PI * 2.0;
            let hammed = dp
                * (A0
                    - ((1.0 - A0)
                        * f32::cos(TAU * i as f32 / FFT_SAMPLES as f32)));
            input_buffer[i] = c32::new(hammed, 0.0);
        }

        plan.c2c(&mut input_buffer, &mut output_buffer)
            .expect("failed fft");

        let fft: Vec<f32> = output_buffer[..FFT_SIZE]
            .iter()
            .map(|elem| {
                // normalise it
                let amplitude = elem.norm_sqr().sqrt();
                amplitude / (FFT_SAMPLES as f32).sqrt()
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
    gl: Glium2d,
    audio_thread: AudioThread,
}

impl App {
    fn new(gl: Glium2d, audio_thread: AudioThread) -> Self {
        Self { gl, audio_thread }
    }

    fn render(&mut self, window: &mut GliumWindow, args: RenderArgs) {
        // we want to let go of that lock as fast as possible, so compute the
        // lines we need to draw and then release it while we draw them
        let vp = args.viewport();
        let (height, width) = (vp.window_size[1], vp.window_size[0]);
        let height = height as f64;
        let width = width as f64;
        let lines: Vec<(Colour, Line)> = {
            let data = self.audio_thread.receiver.read().expect("read lock");

            let fft_lines = data.fft[1..].iter().enumerate().map(|(i, dp)| {
                // TODO resize fft output so we can draw fewer of these
                // lines. we're trying to draw 16k lines in the 1k-ish
                // pixels we have
                let col = rescale(
                    i as f32,
                    (0.0, data.fft.len() as f32 - 1.0),
                    (0.0, width),
                );
                let depth = rescale(*dp, (0.0, 1.0), (0.0, height));
                let line = [col, height, col, height - depth];
                (rgb(0, 0, 255), line)
            });

            // performance hack: we're trying to draw over 10k samples but we
            // don't even have that many pixels. Instead, grab fewer of them.
            // That might mean that we miss all of the negative samples, so draw
            // them symmetrically to compensate
            let samples_step = 10;

            let sample_lines =
                data.samples.iter().step_by(samples_step).enumerate().map(
                    |(i, dp)| {
                        let col = rescale(
                            i as f32,
                            (
                                0.0,
                                data.samples.len() as f32 / samples_step as f32
                                    - 1.0,
                            ),
                            (0.0, width),
                        );
                        let row = rescale(*dp, (0.0, 1.0), (0.0, height / 2.0));
                        let line =
                            [col, height / 2.0 - row, col, height / 2.0 + row];
                        (rgb(0xd8, 0xac, 0x9c), line)
                    },
                );

            let heart_lines = {
                let recent_volume =
                    data.samples[data.samples.len() - RECENT_VOLUME_SAMPLES
                        ..data.samples.len() - 1]
                        .iter()
                        .copied()
                        .fold(0.0, f32::max) as f64;
                let max_volume =
                    data.samples[data.samples.len() - MAX_VOLUME_SAMPLES
                        ..data.samples.len() - 1]
                        .iter()
                        .copied()
                        .fold(0.0, f32::max) as f64;

                let mut ls: Vec<(Colour, Line)> = Vec::new();
                let big_colour = rgb(114, 210, 200);
                let little_colour = rgb(114 / 2, 210 / 2, 200 / 2);

                for (colour, vol) in
                    [(little_colour, recent_volume), (big_colour, max_volume)]
                        .iter()
                {
                    let scale_factor = f64::min(1.0, 2.0 * vol);
                    let margin_x = (
                        width - width * (1.0 - scale_factor) / 2.0,
                        width * (1.0 - scale_factor) / 2.0,
                    );
                    let margin_y = (
                        height - height * (1.0 - scale_factor) / 2.0,
                        height * (1.0 - scale_factor) / 2.0,
                    );
                    let mut heart_shape = parametric(
                        |t| 16.0 * t.sin().powi(3),
                        (-16.0, 16.0),
                        margin_x,
                        |t| {
                            13.0 * t.cos()
                                - 5.0 * (2.0 * t).cos()
                                - 2.0 * (3.0 * t).cos()
                                - (4.0 * t).cos()
                        },
                        (-17.0, 12.0),
                        margin_y,
                        (-PI64, PI64, 0.1),
                        *colour,
                    );
                    ls.append(&mut heart_shape);
                }
                ls
            };

            fft_lines.chain(sample_lines).chain(heart_lines).collect()
        };

        let mut target = window.draw();
        self.gl.draw(&mut target, vp, move |c, gl| {
            const BLACK: Colour = [0.0, 0.0, 0.0, 1.0];
            graphics::clear(BLACK, gl);
            for (colour, line) in lines {
                graphics::line(colour, 1.0, line, c.transform, gl)
            }
        });
        target.finish().unwrap();
    }
}

fn main() -> Result<(), anyhow::Error> {
    let (row_px, col_px) = WINDOW_SIZE;

    let audio_thread = AudioThread::new()?;

    let opengl = match OS {
        "macos" => OpenGL::V3_2,
        _ => OpenGL::V2_1,
    };
    dbg!(&opengl);
    let mut window: GliumWindow = WindowSettings::new(
        "noise",
        [row_px as u32, col_px as u32])
    .exit_on_esc(true)
    .graphics_api(opengl)
    .build().unwrap();

    let mut g2d = Glium2d::new(opengl, &window);
    let mut app = App::new(g2d, audio_thread);

    // handle UI events on the main thread
    while let Some(e) = window.next() {
        if let Some(args) = e.render_args() {
            app.render(&mut window, args);
        }
    }

    Ok(())
}

fn parametric<X, Y>(
    x_fn: X,
    x_expected_range: (f64, f64),
    x_output_range: (f64, f64),
    y_fn: Y,
    y_expected_range: (f64, f64),
    y_output_range: (f64, f64),
    t_range_step: (f64, f64, f64),
    colour: Colour,
) -> Vec<(Colour, Line)>
where
    X: Fn(f64) -> f64,
    Y: Fn(f64) -> f64,
{
    let (start_range, end_range, step) = t_range_step;

    let mut ret =
        Vec::with_capacity(((end_range - start_range) / step).ceil() as usize);

    let mut t = start_range;
    let mut last = None;
    while t < end_range {
        let x = x_fn(t);
        let y = y_fn(t);

        let x = rescale(x, x_expected_range, x_output_range);
        let y = rescale(y, y_expected_range, y_output_range);

        match last {
            None => {
                last = Some((x, y));
            }
            Some((last_x, last_y)) => {
                last = Some((x, y));
                ret.push((colour, [last_x, last_y, x, y]));
            }
        }

        t += step;
    }

    // close the shape
    if ret.len() > 1 {
        let (_colour, [start_x, start_y, _, _]) = ret[0];
        let (_colour, [_, _, end_x, end_y]) = ret[ret.len() - 1];
        ret.push((colour, [start_x, start_y, end_x, end_y]));
    }

    ret
}

fn rescale<F, T>(num: F, from_range: (F, F), to_range: (T, T)) -> T
where
    F: Into<T>,
    T: Float + From<F>,
{
    // get the type conversions out of the way
    let (from_min, from_max) = from_range;
    let (from_min, from_max): (T, T) = (from_min.into(), from_max.into());
    let num: T = num.into();

    let from_span: T = from_max - from_min;
    let percent: T = (num - from_min) / from_span;
    let (to_min, to_max) = to_range;
    let to_span: T = to_max - to_min;
    let output = percent * to_span + to_min;
    output
}

#[test]
fn test_rescale() {
    assert_eq!(rescale(5u8, (0u8, 10u8), (-1f32, 1f32)), 0.0);
    assert_eq!(rescale(0.5, (-1f32, 1f32), (0f32, 1f32)), 0.75);
    assert_eq!(rescale(0.5, (-1f32, 1f32), (0f32, 200f32)), 150f32);
    assert_eq!(rescale(0.25, (-1f32, 1f32), (1f32, -1f32)), -0.25);
}

fn rgb(r: u8, g: u8, b: u8) -> Colour {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
}
