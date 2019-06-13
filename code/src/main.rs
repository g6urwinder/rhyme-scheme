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
        let mut rng = rand::thread_rng();
        
        if layers_sizes.len() < 2 {
            panic!("must be at least two layers");
        } 

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("can't have any empty layers");
            }
        }

        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();

        let first_layer_size = *it.next().unwrap();
        let mut prev_layer_size = first_layer_size;
        
        //println!("FIRST LAYER SIZE {:?}", first_layer_size);
        //println!("PREV LAYER SIZE {:?}", prev_layer_size);

        for &layer_size in it {
            let mut layer: Vec<Vec<f64>> = Vec::new();
            
            //println!("LAYER => {:?}", layer);
            for z in 0..layer_size {
                
                let mut node: Vec<f64> = Vec::new();
                for x in 0..prev_layer_size+1 {
                    let random_weight: f64 = rng.gen_range(-0.5f64, 05f64);
                    node.push(random_weight);

                    //println!("NODE AFTER PUSHING RANDOM WEIGHT {:?} => {:?} => {:?}",x,  prev_layer_size, node);
                }
                node.shrink_to_fit();

                //println!("NODE AFTER SHRINKING {:?} => {:?} => {:?}", z, layer_size,  node);
                layer.push(node)
            }
            layer.shrink_to_fit();

            //println!("LATER AFTER SHRINKING {:?}", layer);
            layers.push(layer);
            prev_layer_size = layer_size;

            //println!("PREVIOUS_LAYER_SIZE_AT_END:: {:?}", prev_layer_size);
        }

        layers.shrink_to_fit();

        NN { layers : layers, num_inputs: first_layer_size }
    }

    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        
        if inputs.len() as u32 != self.num_inputs {
            panic!("input has a different length than the network's input layer");
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
    
    pub fn to_json(&self) -> String {
        json::encode(self).ok().expect("encoding JSON failed")
    }

    pub fn from_json(encoded: &str) -> NN {
        let network: NN = json::decode(encoded).ok().expect("decoding JSON failed");
        network
    }

    fn train_details(&mut self, 
                     examples: &[(Vec<f64>, Vec<f64>)], 
                     rate: f64, 
                     momentum: f64,
                     log_interval: Option<u32>,
                     halt_condition: HaltCondition) -> f64 {
    
        let input_layer_size = self.num_inputs;
        let output_layer_size = self.layers[self.layers.len()-1].len();

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

    fn train_incremental(
        &mut self,
        examples: &[(Vec<f64>, Vec<f64>)],
        rate: f64,
        momentum: f64,
        log_interval: Option<u32>,
        halt_condition: HaltCondition
        ) -> f64 {
    
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
                    if epochs == epochs_halt { break }
                },
                MSE(target_error) => {
                    if training_error_rate <= target_error { break }
                },
                Timer(duration) => {
                    let now = PreciseTime::now();
                    if start_time.to(now) >= duration { break }
                }
            }
           }

           training_error_rate = 0f64;

           for &(ref inputs, ref targets) in examples.iter() {
            let results = self.do_run(&inputs);
            let weight_updates = self.calculate_weight_updates(&results, &targets);
            training_error_rate += calculate_error(&results, &targets);
            self.update_weights(&weight_updates, &mut prev_deltas, rate, momentum)
           } 

           epochs += 1;
       }

       training_error_rate
    }

    fn do_run(&self, _: &[f64]) -> Vec<Vec<f64>> {
        
        Vec::new()
    }

    fn update_weights(
        &mut self,
        _: &Vec<Vec<Vec<f64>>>,
        _: &mut Vec<Vec<Vec<f64>>>,
        _: f64,
        _: f64) {
    
    }

    fn calculate_weight_updates(&self, _: &Vec<Vec<f64>>, _: &[f64]) -> Vec<Vec<Vec<f64>>> {
        
        Vec::new()
    }

    fn make_weights_tracker<T: Clone>(&self, _: T) -> Vec<Vec<Vec<T>>> {
        
        Vec::new()
    }

}
fn calculate_error(_: &Vec<Vec<f64>>, _: &[f64]) -> f64 {
        
    0f64
}


#[cfg(test)]
mod tests {
    
    use super::*;

    #[test]
    fn superman() {
        let examples = [
                (vec![0f64, 0f64], vec![0f64]),
                (vec![0f64, 1f64], vec![1f64]),
                (vec![1f64, 0f64], vec![1f64]),
                (vec![1f64, 1f64], vec![0f64]),
            ];

        let mut net1 = NN::new(&[2,4,3,1]);
        println!("{:?}", net1)
    }
}

fn main() {
    println!("Hello, world!");
}
