extern crate rand;
extern crate rustc_serialize;
extern crate time;
use HaltCondition::{ Epochs, MSE, Timer };

use LearningMode::{ Incremental };
use time::{ Duration, PreciseTime };
use rand::Rng;
use rustc_serialize::json;
use std::slice;

static DEFAULT_LEARNING_RATE: f64 = 0.3f64;
static DEFAULT_MOMENTUM: f64 = 0f64;
static DEFAULT_EPOCHS: u32 = 1000;

#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    Epochs(u32), /// stop training after a certain number of Epochs
    MSE(f64), /// Train until a certain error rate is achieved
    Timer(Duration),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearningMode {
    /// train the network Incrementally (updates weights after each example)
    Incremental
}

/// used to specify options that tells how a network will be trained
#[derive(Debug)]
pub struct Trainer<'a,'b> {
    examples: &'b [(Vec<f64>, Vec<f64>)],
    rate: f64,
    momentum: f64,
    log_interval: Option<u32>,
    halt_condition: HaltCondition,
    learning_mode: LearningMode,
    nn: &'a mut NN,
}
/// trainer is used to chain together options that specify how to train a network.
/// All of the options are optional because the Trainer struct has default values for each option.
/// However the go() method must be called or the network will not be trained.

impl<'a, 'b> Trainer<'a, 'b> {

    /// specifies the learning rate to be used when training (default value is 0.3)
    /// this is the step size used in backpropagation algorithm.
    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a, 'b> {
        if rate  <= 0f64 {
            panic!(" the learning rate must be a positive number");
        }

        self.rate = rate;
        self
    }

    /// specifies the momentum to be used when training (default value is 0.0)
    /// Note :- need to understand why default value is 0.0 for momentum
    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a, 'b> {
        if momentum <= 0f64 {
            panic!(" momentum must be positive number");
        }

        self.momentum = momentum;
        self
    }

    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a, 'b> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!(" log interval must be some positive number or None")
            }
            _ => ()
        }

        self.log_interval = log_interval;
        self
    }

    /// specifies when to stop training. 'Epochs(x)' will stop the training after
    /// 'x' epochs (one epoch is one loop through all of the training examples)
    /// while 'MSE(e)' will stop the training when the error rate
    /// is or below  'e'. 'Timer(d)' will halt after the [duration] 'd' has elapsed

    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a, 'b> {
        match halt_condition {
            Epochs(epochs) if epochs < 1 => {
                panic!("must train for at least one epoch")
            }
            MSE(mse) if mse <= 0f64 => {
                panic!("MSE must be greater than 0")
            }
            _ => ()
        }

        self.halt_condition = halt_condition;
        self
    }

    /// specifies what mode to train the network in.
    /// incremental means to increase the weights in the network after every example.
    pub fn learning_mode(&mut self, learning_mode: LearningMode ) -> &mut Trainer<'a, 'b> {
        self.learning_mode = learning_mode;
        self
    }

    pub fn go(&mut self) -> f64 {
        self.nn.train_details(
            self.examples,
            self.rate,
            self.momentum,
            self.log_interval,
            self.halt_condition
        )
    }
}

/// Neural network
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct NN {
        layers: Vec<Vec<Vec<f64>>>,
        num_inputs: u32,
    }

impl NN {
    pub fn new(layers_sizes: &[u32]) -> NN {
        let mut rng = rand::thread_rng();

        if layers_sizes.len() < 2 {
            panic!(" must have at least 2 layers");
        }

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("Cant have any empty layers");
            }
        }

        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();

        let first_layer_size = *it.next().unwrap();

        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_size {
                    let mut node: Vec<f64> = Vec::new();
            for _ in 0..prev_layer_size+1 {
                let random_weight: f64 = rng.gen_range(-0.5f64, 0.5f64);
                node.push(random_weight);
               }
               node.shrink_to_fit();
               layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        NN {layers: Vec::new(), num_inputs: 0u32}
    }

fn train_details(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
               halt_condition: HaltCondition) -> f64 {
   let input_layer_size = self.num_inputs;
   let output_layer_size = self.layers[self.layers.len() - 1].len();
   for &(ref inputs, ref outputs) in examples.iter() {
       if inputs.len() as u32 != input_layer_size {
           panic!("input has a different length than the network's input layer");
       }
       if outputs.len() != output_layer_size {
           panic!("output has a different length than the network's output layer");
       }
   }

   self.train_incremental(examples, rate, momentum, log_interval, halt_condition)
}

fn train_incremental(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition) -> f64 {
    let mut prev_deltas = self.make_weights_tracker(0.0f64);
    let mut epochs = 0u32;
    let mut training_error_rate = 0f64;
    let start_time = PreciseTime::now();

    loop {
        if epochs > 0 {
            match log_interval {
                Some(interval) if epochs % interval == 0 => {
                    println!("error rate: {}", training_error_rate);
                },

                _ => (),
            }

            match halt_condition {
                Epochs(epochs_halt) => {
                    if epochs == epochs_halt {break}
                },
                MSE(target_error) => {
                    if training_error_rate <= target_error {break}
                },
                Timer(duration) => {
                    let now = PreciseTime::now();
                    if start_time.to(now) >= duration {break}
                }
            }
        }

        training_error_rate = 0f64;

        for &(ref inputs, ref targets) in examples.iter() {
            let results = self.do_run(&inputs);
            let weight_updates = self.calculate_weight_updates(&results, &targets);
            training_error_rate += calculate_error(&results, &targets);
            self.update_weights(weight_updates, &mut prev_deltas, rate, momentum)
        }

        epochs += 1;
    }
    training_error_rate
}

    fn calculate_weight_updates(&self, _: &Vec<Vec<f64>>, _: &[f64]) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }

    fn make_weights_tracker<T: Clone>(&self, _: T) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }

    fn update_weights(&mut self, _: Vec<Vec<Vec<f64>>>, _: &mut Vec<Vec<Vec<f64>>>, _: f64, _:f64) {

    }

    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() as u32 != self.num_inputs {
            panic!(" input has a different length than the networks input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }

   pub fn train<'b>(&'b mut self, examples: &'b [(Vec<f64>, Vec<f64>)]) -> Trainer {
       Trainer {
           examples: examples,
           rate: DEFAULT_LEARNING_RATE,
           momentum: DEFAULT_MOMENTUM,
           log_interval: None,
           halt_condition: Epochs(DEFAULT_EPOCHS),
           learning_mode: Incremental,
           nn: self
       }
   }

   pub fn do_run(&self, _: &[f64]) -> Vec<Vec<f64>> {
       Vec::new()
   }

   pub fn to_json(&self) -> String {
       json::encode(self).ok().expect("encoding JSON failed")
   }

   pub fn from_json(encoded: &str) -> NN {
       let network: NN = json::decode(encoded).ok().expect("decoding JSON failed");
       network
   }

}

fn calculate_error(_ : &Vec<Vec<f64>>, _: &[f64]) -> f64 {
    0f64
}

fn main() {
    println!("Hello, world!");
}
