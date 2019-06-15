extern crate rand;

use rand::Rng;

#[derive(Debug, Copy, Clone)]
pub struct Unit {
    pub value: f64,
    pub gradient: f64,
}

impl Unit {
    
    pub fn empty() -> Self {
        Unit {
            value: 0.0,
            gradient: 0.0,
        }
    }

    pub fn new(value: f64, gradient: f64) -> Self {
        Unit {
            value: value,
            gradient: gradient,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MGate {    
    pub u0: Unit,
    pub u1: Unit,
    pub output: Unit,
}

impl MGate {
    
    pub fn new() -> Self {
        MGate {
            u0: Unit::empty(), 
            u1: Unit::empty(),
            output: Unit::empty(),
        }
    }

    pub fn forward(mut self, u0: Unit, u1: Unit) -> Self {
        
        self.u0 = u0;
        self.u1 = u1;
        self.output = Unit::new(self.u0.value * self.u1.value, 0.0);
        self
    }

    pub fn backward(mut self) -> Self {
        
        self.u0.gradient += self.u1.value * self.output.gradient;
        self.u1.gradient += self.u0.value * self.output.gradient;
        self
    }
}

#[derive(Debug, Copy, Clone)]
pub struct AGate {
    pub u0: Unit,
    pub u1: Unit,
    pub output: Unit,
}

impl AGate {
    
    pub fn new() -> Self {
        AGate {
            u0: Unit::empty(),
            u1: Unit::empty(),
            output: Unit::empty(),
        }
    }

    pub fn forward(mut self, u0: Unit, u1: Unit) -> Self {
            
        self.u0 = u0;
        self.u1 = u1;
        self.output = Unit::new(self.u0.value + self.u1.value, 0.0);
        self
    }

    pub fn backward(mut self) -> Self {
        
        self.u0.gradient += 1.0 * self.output.gradient;
        self.u1.gradient += 1.0 * self.output.gradient;
        self
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SigmoidGate {
    
    pub u0: Unit,
    pub u1: Unit,
    pub output: Unit,
}

impl SigmoidGate {
    
    pub fn new() -> Self {
        
        SigmoidGate {
            u0: Unit::empty(),
            u1: Unit::empty(),
            output: Unit::empty(),
        }
    }

    pub fn forward(mut self, u0: Unit) -> Self {
        
        self.u0 = u0;
        self.output = Unit::new(sig(self.u0.value), 0.0);
        self
    }

    pub fn backward(mut self) -> Self {
        
        let s = sig(self.u0.value);
        self.u0.gradient += (s*(1.0-s))*self.output.gradient;
        self
    }
}

pub fn sig(x: f64) -> f64 {
        
    if x > 0.0 {
        1.0/(1.0 + (1.0/dopow(x)))
    } else {
        1.0/(1.0 + dopow(-1.0*x)) as f64
    }
}


pub fn dopow(exp: f64) -> f64 {
    std::f64::consts::E.powi(exp as i32)
}

#[derive(Debug, Copy, Clone)]
pub struct Circuit {
    pub ax: MGate,
    pub by: MGate,
    pub axpby: AGate,
    pub axpbypc: f64,
    pub mulg0: MGate,
    pub mulg1: MGate,
    pub addg0: AGate,
    pub addg1: AGate
}

impl Circuit {
    
    pub fn new() -> Self {
        
        Circuit {
            ax: MGate::new(),
            by: MGate::new(),
            axpby: AGate::new(),
            axpbypc: 0.0,
            mulg0: MGate::new(),
            mulg1: MGate::new(),
            addg0: AGate::new(),
            addg1: AGate::new(),
        }
    }

    pub fn forward(mut self, x: Unit, y: Unit, a: Unit, b: Unit, c: Unit) -> Self {
        
        self.ax = MGate::forward(self.mulg0, a, x);
        self.by = MGate::forward(self.mulg1, b, y);
        self.axpby = AGate::forward(self.addg0, self.ax.output, self.by.output);
        self.axpbypc = AGate::forward(self.addg1, self.axpby.output, c).output.value;
        self
    }

    pub fn backward(mut self, gradient_output: f64) -> Self {
        
        self.axpbypc = gradient_output;
        self.addg1 = AGate::backward(self.addg1);
        self.addg0 = AGate::backward(self.addg0);
        self.mulg1 = MGate::backward(self.mulg1);
        self.mulg0 = MGate::backward(self.mulg0);
        self
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SVM {
    pub a: Unit,
    pub b: Unit,
    pub c: Unit,
    pub circuit: Circuit,
    pub unit_out: f64,
}

impl SVM {
    
    pub fn new() -> Self {
        
        SVM {
            
            a: Unit::new(1.0, 0.0),
            b: Unit::new(-2.0, 0.0),
            c: Unit::new(-1.0, 0.0),
            circuit: Circuit::new(),
            unit_out: 0.0,
        }
    }

    pub fn forward(mut self, x: Unit, y: Unit) -> Self {
       
        self.circuit = Circuit::forward(self.circuit, x, y, self.a, self.b, self.c);
        self.unit_out = self.circuit.axpbypc;
        self
    }

    pub fn backward(mut self, label: i8) -> Self {
        
        self.a.gradient = 0.0;
        self.b.gradient = 0.0;
        self.c.gradient = 0.0;

        let mut pull = 0.0;
        if (label == 1 && self.unit_out < 1.0) {
            pull = 1.0;
        }
        if (label == -1 && self.unit_out > -1.0) {
            pull = -1.0;
        }

        self.circuit = Circuit::backward(self.circuit, pull);
        self.a.gradient += -self.a.value;
        self.b.gradient += -self.b.value;

        self
    }

    pub fn learn_from(mut self, x: Unit, y: Unit, label: i8) -> Self {
        
        self = SVM::forward(self, x, y);
        self = SVM::backward(self, label);
        self = SVM::parameter_update(self);

        self
    }

    pub fn parameter_update(mut self) -> Self {
    
        let step_size = 0.011;
        self.a.value += step_size * self.a.gradient;
        self.b.value += step_size * self.b.gradient;
        self.c.value += step_size * self.c.gradient;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn eval_training_accuracy(mut data: Vec<Vec<f64>>,mut labels: Vec<i8>, mut svm: SVM) -> (SVM, f64) {
        
        let mut num_correct = 0.0 as f64;

        for i in 0..data.len() {
            
            let x = Unit::new(data[i][0], 0.0);
            let y = Unit::new(data[i][1], 0.0);
            let true_label = labels[i];

            let mut predicted_level = 1;
            svm = SVM::forward(svm, x, y);
            if svm.unit_out > 0.0 {
                predicted_level = 1;
            } else {
                predicted_level = -1;
            }

            if predicted_level == true_label {
                num_correct = num_correct + 1.0;
            }
        }

        (svm, num_correct/data.len() as f64)
    }
    
    #[derive(Debug, Clone)]
    pub struct DataLabel {
        pub data: Vec<Vec<f64>>,
        pub labels: Vec<i8>,
    }

    impl DataLabel {
        
        pub fn new() -> Self {
            
            DataLabel {
                data: Vec::new(),
                labels: Vec::new(),
            }
        }

        pub fn push(mut self, data: Vec<f64>, labels: i8) -> Self {
           
            self.data.push(data);
            self.labels.push(labels);
            DataLabel {
                data: self.data,
                labels: self.labels,
            }

        }

        pub fn push_data(self, data: Vec<Vec<f64>>) -> Self {
            
            DataLabel {
                data: data,
                labels: self.labels,
            }
        }

        pub fn push_labels(self, labels: Vec<i8>) -> Self {
            
            DataLabel {
                data: self.data,
                labels: labels,
            }
        }
    }

    #[test]
    fn test_svm() {
       
        let mut data_labels = DataLabel::new();
        let mut v1 = Vec::new(); v1.push(1.2); v1.push(0.7); 
        data_labels = DataLabel::push(data_labels.clone(), v1, 1 );
        
        let mut v2 = Vec::new(); v2.push(-0.3); v2.push(-0.5);
        data_labels = DataLabel::push(data_labels.clone(), v2, -1);

        let mut v3 = Vec::new(); v3.push(3.0); v3.push(0.1);
        data_labels = DataLabel::push(data_labels.clone(), v3, 1);

        let mut v4 = Vec::new(); v4.push(-0.1); v4.push(-1.0);
        data_labels = DataLabel::push(data_labels.clone(), v4, -1);

        let mut v5 = Vec::new(); v5.push(-1.0); v5.push(1.1);
        data_labels = DataLabel::push(data_labels.clone(), v5, -1);

        let mut v6 = Vec::new(); v6.push(2.1); v6.push(-3.0);
        data_labels = DataLabel::push(data_labels.clone(), v6, 1);
 
        let mut svm = SVM::new();

       for iter in 0..400 {
        
           let mut i = rand::thread_rng().gen_range(0, data_labels.clone().data.len()) as usize;
           let mut x = Unit::new(data_labels.clone().data[i][0], 0.0);
           let mut y = Unit::new(data_labels.clone().data[i][1], 0.0);

           let mut label = data_labels.clone().labels[i];

           svm = SVM::learn_from(svm, x, y, label);

           if (iter % 10 == 0) {
               let (svm_new, acc) = eval_training_accuracy(data_labels.clone().data, data_labels.clone().labels, svm); 
               svm = svm_new.clone();
               println!("TRAINING ACCURACY AT ITER {:?} : {:?}", iter, acc)
           }
       } 
    }
}
