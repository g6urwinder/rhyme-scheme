extern crate rand;
extern crate rustc_serialize;
extern crate time;
use HaltCondition::{ Epochs, MSE, Timer };

use LearningMode::{ Incremental };
use time::{ Duration, PreciseTime };
use std::iter::{Zip, Enumerate};
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
            for z in 0..layer_size {
            let mut node: Vec<f64> = Vec::new();
            for x in 0..prev_layer_size+1 {
                let random_weight: f64 = rng.gen_range(-0.5f64, 05f64);
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
        NN {layers: layers, num_inputs: first_layer_size}
    }

    fn train_details(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition) -> f64 {

        // check that input and output sizes are correct
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
                    // log error rate if necessary
                    match log_interval {
                        Some(interval) if epochs % interval == 0 => {
                            println!("error rate: {}", training_error_rate);
                        },
                        _ => (),
                    }

                    // check if we've met the halt condition yet
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

    fn calculate_weight_updates(&self, results: &Vec<Vec<f64>>, targets: &[f64]) -> Vec<Vec<Vec<f64>>> {
        let mut network_errors: Vec<Vec<f64>> = Vec::new();
        let mut network_weight_updates = Vec::new();
        let layers = &self.layers;
        let network_results = &results[1..];
        let mut next_layer_nodes: Option<&Vec<Vec<f64>>> = None;

        for (layer_index, (layer_nodes, layer_results)) in iter_zip_enum(layers, network_results).rev() {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::new();
            let mut layer_weight_updates = Vec::new();


            for (node_index, (node, &result)) in iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::new();
                let mut node_error;


                if layer_index == layers.len() - 1 {
                    node_error = result * (1f64 - result) * (targets[node_index] - result);
                } else {
                    let mut sum = 0f64;
                    let next_layer_errors = &network_errors[network_errors.len() -  1];
                    for (next_node, &next_node_error_data) in next_layer_nodes.unwrap().iter().zip((next_layer_errors).iter()) {
                        sum += next_node[node_index + 1] * next_node_error_data;
                    }
                    node_error = result * (1f64 - result) * sum;
                }

                for weight_index in 0..node.len() {
                    let mut prev_layer_result;
                    if weight_index == 0 {
                        prev_layer_result = 1f64;
                    } else {
                        prev_layer_result = prev_layer_results[weight_index -1];
                    }
                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        network_weight_updates.reverse();
        network_weight_updates
     }

    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new();
        for layer in self.layers.iter() {
            let mut layer_level = Vec::new();
            for node in layer.iter() {
                let mut node_level = Vec::new();
                for _ in node.iter() {
                    node_level.push(place_holder.clone());
                }
                layer_level.push(node_level);
            }
            network_level.push(layer_level);
        }
        network_level
    }

    fn update_weights(&mut self, network_weight_updates: &Vec<Vec<Vec<f64>>>, prev_deltas: &mut Vec<Vec<Vec<f64>>>, rate: f64, momentum: f64) {
      for layer_index in 0..self.layers.len() {
          let mut layer = &mut self.layers[layer_index];
          let layer_weight_updates = &network_weight_updates[layer_index];
          for node_index in 0..layer.len() {
              let mut node = &mut layer[node_index];
              let node_weight_updates = &layer_weight_updates[node_index];
              for weight_index in 0..node.len() {
                  let weight_update = node_weight_updates[weight_index];
                  let prev_delta = prev_deltas[layer_index][node_index][weight_index];
                  let delta = (rate * weight_update) + (momentum * prev_delta);
                  node[weight_index] += delta;
                  prev_deltas[layer_index][node_index][weight_index] = delta;
              }
          }
      }

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

   pub fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
       let mut results = Vec::new();
       results.push(inputs.to_vec());
       for(layer_index, layer) in self.layers.iter().enumerate() {
           let mut layer_results = Vec::new();
           for node in layer.iter() {
               layer_results.push( sigmoid(modified_dotprod(&node, &results[layer_index])))
           }

           results.push(layer_results);
       }
       results
   }

   pub fn to_json(&self) -> String {
       json::encode(self).ok().expect("encoding JSON failed")
   }

   pub fn from_json(encoded: &str) -> NN {
       let network: NN = json::decode(encoded).ok().expect("decoding JSON failed");
       network
   }

}

fn calculate_error(results : &Vec<Vec<f64>>, targets: &[f64]) -> f64 {
    let ref last_results = results[results.len() -1];
    let mut total:f64 = 0f64;
    for(&result, &target) in last_results.iter().zip(targets.iter()) {
        total += (target - result).powi(2);
    }
    total / (last_results.len() as f64)
}

fn iter_zip_enum<'s, 't, S: 's, T: 't>(s: &'s [S], t: &'t [T]) ->
    Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>>  {
    s.iter().zip(t.iter()).enumerate()
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
        println!("{:?}", net1);

        net1.train(&examples)
            .log_interval(Some(1000))
            .halt_condition(HaltCondition::MSE(0.01))
            .learning_mode( LearningMode::Incremental)
            .momentum(0.5)
            .rate(0.5)
            .go();

        for &(ref inputs, ref outputs) in examples.iter() {

            let results = net1.run(inputs);

            println!("GAVE INPUTS {:?}, TO NET {:?}, GAVE RESULTS {:?}", inputs, net1, results);

            let (result, key) = (results[0].round(), outputs[0]);

            println!("RESULT {:?} WITH KEY {:?}", result, key);
        }
    }
}

fn sigmoid(y: f64) -> f64 {
    1f64 / (1f64 + (-y).exp())
}

fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap();

    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

fn main() {
    println!("Hello, world!");
}
