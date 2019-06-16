extern crate serde;
extern crate serde_json;

extern crate rusty_machine;
extern crate rustc_serialize;
use rustc_serialize::json::Json;
use std::fs::File;
use std::io::Read;

use rusty_machine::learning::svm::SVM;
use rusty_machine::learning::SupModel;
use rusty_machine::learning::toolkit::kernel::HyperTan;

use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

fn typeid(_: f64) {
    println!("{:?}", 2);
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Sample {
    rms: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug,  Clone)]
struct JSOND {
    samples: Vec<Sample>,
}

fn train() {
    
    println!("Training SVM ...");
    let mut file = File::open("/home/ridhmbusiness/rhyme-scheme/dataset/numbers_wav/5/5515.json").unwrap();

    let mut data = String::new();
    file.read_to_string(&mut data);
    
    let mut json: JSOND = serde_json::from_str(&data).unwrap();
    
    let mut samples: Vec<Sample> = json.clone().samples;
    
    println!("PRINTING TYPE JSOND {:?}", json.clone());
    //typeid(samples);

    let mut inputs_vec = Vec::new();
    let mut targets_vec = Vec::new();
    for x in 0..samples.len() {
        if samples[x].rms == None {
        
        } else {
            inputs_vec.push(samples[x].rms.unwrap_or(0.0));
            targets_vec.push(1.);
        }
    } 

    let inputs = Matrix::new(targets_vec.len(), 1, inputs_vec);

    let targets = Vector::new(targets_vec);

    println!("INPUTS VECTOR {:?}", inputs);

    let mut svm = SVM::new(HyperTan::new(100., 0.), 0.3);
    svm.train(&inputs, &targets).unwrap();

    println!("Evaluation ...");

    let mut hits = 0;
    let mut misses = 0;
    
    for n in (-1000..1000).filter(|&x| x % 100 == 0) {
        
        let nf = n as f64;
        let input = Matrix::new(1, 1, vec![nf]);
        let out = svm.predict(&input).unwrap();

        let res = if out[0]*nf > 0. {
            
            hits += 1;
            true
        } else if nf == 0. {
            
            hits += 1;
            true
        } else {
            
            misses += 1;
            false
        };

        println!("{} => {} : {}", Matrix::data(&input)[0], out[0], res);
    }

    println!("Performance report :");
    println!("HiTS {}, Misses: {}", hits, misses);
    let hits_f = hits as f64;
    let total = (hits + misses) as f64;
    println!("Accuracy {}, ", (hits_f/ total) * 100.);
}

#[cfg(test)]
mod tests {
    
    use super::*;

    #[test]
    fn test() {
        
        train();
    }
}
