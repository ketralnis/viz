use anyhow;
use cpal;
use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{StreamData, UnknownTypeInputBuffer};
use glutin_window::GlutinWindow as Window;
use graphics::{self, Transformed};
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rustfft::num_complex::Complex32;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::thread;

const GRAPHICS_HZ: u64 = 15;
const WINDOW_SIZE: (usize, usize) = (200, 200);
const PIXEL_SIZE: u32 = 2;
const NUM_SAMPLES: usize = 16384;

struct AudioThread {
    receiver: Receiver<Vec<f32>>,
}

impl AudioThread {
    pub fn new() -> Result<Self, anyhow::Error> {
        let (producer, receiver) = sync_channel(1);

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no default audio device");
        let event_loop = host.event_loop();
        let format = device.default_input_format()?;
        let _stream_id = event_loop.build_input_stream(&device, &format);
        let mut rb: Vec<Complex32> = Vec::with_capacity(NUM_SAMPLES);
        let mut planner: FFTplanner<f32> = FFTplanner::new(false);
        let fft = planner.plan_fft(NUM_SAMPLES);

        thread::spawn(move || {
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

                let bytes: Vec<f32> = match stream_data {
                    StreamData::Input {
                        buffer: UnknownTypeInputBuffer::U16(buf),
                    } => buf.iter().map(|b| *b as f32).collect(),
                    StreamData::Input {
                        buffer: UnknownTypeInputBuffer::I16(buf),
                    } => buf.iter().map(|b| *b as f32).collect(),
                    StreamData::Input {
                        buffer: UnknownTypeInputBuffer::F32(buf),
                    } => buf.iter().map(|b| *b as f32).collect(),
                    StreamData::Output { .. } => {
                        unreachable!("got output data?")
                    }
                };
                for b in bytes {
                    rb.push(Complex32::new(b, 0.0));

                    if rb.len() >= NUM_SAMPLES {
                        let mut fft_output: Vec<Complex32> =
                            vec![Zero::zero(); NUM_SAMPLES];
                        fft.process(&mut rb, &mut fft_output);
                        let as_reals: Vec<f32> =
                            fft_output.iter().map(|cm| cm.re).collect();
                        producer.send(as_reals).expect("failed to send");
                        rb.truncate(0);
                    }
                }
            });
        });

        Ok(Self { receiver })
    }
}

pub struct App {
    gl: GlGraphics,
    audio_thread: AudioThread,
    fft_output: Option<Vec<f32>>,
}

impl App {
    fn new(gl: GlGraphics, audio_thread: AudioThread) -> Self {
        Self {
            gl,
            audio_thread,
            fft_output: None,
        }
    }

    fn render(&mut self, args: RenderArgs) {
        let fft_output = &self.fft_output.as_ref();
        self.gl.draw(args.viewport(), |c, gl| {
            const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

            graphics::clear(BLACK, gl);

            let square =
                graphics::rectangle::square(0.0, 0.0, PIXEL_SIZE as f64);

            let (rows, cols) = WINDOW_SIZE;

            match fft_output {
                Some(data) if data.len() > 0 => {
                    let samples = data.len() as f32;
                    let max =
                        data.iter().cloned().fold(0.0, f32::max) / samples;
                    let min =
                        data.iter().cloned().fold(0.0, f32::min) / samples;

                    for (i, dp) in data.iter().enumerate() {
                        let row = (i / rows) as u32 / PIXEL_SIZE;
                        let col = (i % cols) as u32 / PIXEL_SIZE;

                        let transform = c.transform.trans(
                            col as f64 * PIXEL_SIZE as f64,
                            row as f64 * PIXEL_SIZE as f64,
                        );
                        let blueness = *dp;
                        let colour = [0.0, 0.0, blueness, 1.0];
                        graphics::rectangle(colour, square, transform, gl);
                    }
                }

                // we don't have data
                _ => (),
            }
        });
    }

    fn update(&mut self, _args: UpdateArgs) {
        loop {
            match self.audio_thread.receiver.try_recv() {
                Ok(data) => self.fft_output = Some(data),
                Err(TryRecvError::Empty) => {
                    // it's not ready for us
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    panic!("audio thread hung up")
                }
            }
        }
    }
}

fn main() -> Result<(), anyhow::Error> {
    println!("Hello, world!");

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
    settings.max_fps = GRAPHICS_HZ;
    settings.ups = GRAPHICS_HZ;
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

fn rescale(num: f32, from_range: (f32, f32), to_range: (f32, f32)) -> f32 {
    let (from_min, from_max) = from_range;
    let (to_min, to_max) = to_range;

    let from_span = from_max - from_min;
    let percent = (num - from_min) / from_span;

    let to_span = to_max - to_min;
    percent * to_span + to_min
}

/*hamming_window
for(int i = 0; i < SEGMENTATION_LENGTH;i++){
    timeDomain[i] = (float) (( 0.53836 - ( 0.46164 * Math.cos( TWOPI * (double)i  / (double)( SEGMENTATION_LENGTH - 1 ) ) ) ) * frameBuffer[i]);
}*/
