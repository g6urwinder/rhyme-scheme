extern crate rand;
extern crate rustc_serialize;
extern crate time;

use HaltCondition::{ Epochs, MSE, Timer};
use LearningMode:: { Incremental };
use std::iter::{Zip, Enumerate};
use std::slice;
use rustc_serialize::json;
use time::{ Duration, PreciseTime};
use rand::Rng;

static DEFAULT_LEARNING_RATE: f64 = 0.3f64;
static DEFAULT_MOMENTUM: f64 = 0f64;
static DEFAULT_EPOCHS: u32 = 1000;

#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    Epochs(u32), // Stop training after a certain number of epochs
    MSE(f64), // Train until a certain error rate is achieved
    Timer(Duration), // Train for fixed amount of time and then halt
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearningMode {
    Incremental // train for network Incrementally ( update weights after each example)
}

#[derive(Debug)]
pub struct Trainer<'a, 'b> {
    examples: &'b [(Vec<f64>, Vec<f64>)],
    rate: f64,
    momentum: f64,
    log_interval: Option<u32>,
    halt_condition: HaltCondition,
    learning_mode: LearningMode,
    nn: &'a mut NN,
}

impl<'a, 'b> Trainer<'a, 'b> {
   
    /*
     * DEFAULT 0.3
     */
    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a, 'b> {
        if rate <= 0f64 {
            panic!("the learning rate must be a positive number");
        }
        
        self.rate = rate;
        self
    }

    /*
     * DEFAULT: 0.0
     */
    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a, 'b> {
        if momentum <= 0f64 {
         panic!("momentum must be positive");   
        }

        self.momentum = momentum;
        self
    }
    
    /*
     * Specifies how often (measured in batches) to log the current rate
     * (mean squared error) during training. `Some(x)` means log after every
     * `x` batches and `None` menas never log
     */
    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a, 'b> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!("log interval must be Some positive number or None")
            }
            _ => ()
        }

        self.log_interval = log_interval;
        self
    }
    
    /*
     * Specifies when to stop training
     * `Epochs(x)` will stop training after `x` epochs
     * `MSE(e)` will stop training when the error rate is at or below `e`
     * `Timer(d)` will halt after the elapsed
     */
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a, 'b> {
        match halt_condition {
            Epochs(epochs) if epochs < 1 => {
                panic!("must train for atleast one epoch")
            }
            MSE(mse) if mse <= 0f64 => {
                panic!("MSE must be greater than 0")
            }
            _ => ()
        }

        self.halt_condition = halt_condition;
        self
    }

    /*
     * Specifies what [mode] to train the network in
     * `Incremental` means to update the weights in the network after every example.
     */
    pub fn learning_mode(&mut self, learning_mode: LearningMode) -> &mut Trainer<'a, 'b> {
        self.learning_mode = learning_mode;
        self
    }

    /*
     * When `go` is called, the network will being training based on the
     * Options specified.
     */
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

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct NN {
    layers: Vec<Vec<Vec<f64>>>,
    num_inputs: u32,
}

impl NN {
    
    /*
     * Each number in the `layers_sizes` parameter specifies a
     * layer in the network. The number itself is the number of nodes in that
     * layer. The first number is the input layer, the last number is the
     * output layer, and all numbers between the first and last are
     * hidden layers.There must be at least two layers in the network.
     */
    pub fn new(layers_sizes: &[u32]) -> NN {
        //let mut _ = rand::thread_rng();
        
        if layers_sizes.len() < 2 {
            panic!("must be at least two layers");
        } 

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("can't have any empty layers");
            }
        }

        NN { layers : Vec::new(), num_inputs: 0u32 }
    }

    fn train_details(&mut self, _: &[(Vec<f64>, Vec<f64>)], _: f64, _: f64, _: Option<u32>, _: HaltCondition) -> f64  {
        
        0f64
    }
}

fn main() {
    println!("Hello, world!");
}