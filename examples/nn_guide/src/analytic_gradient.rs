extern crate rand;
use rand::Rng;

mod foo {

    #[derive(Debug, Copy, Clone)]
    pub struct Unit {
        pub value: f64,
        pub grad: f64,
    }

    impl Unit {
        pub fn empty() -> Unit {
            Unit {
                value: 0.0,
                grad: 0.0,
            }
        }

        pub fn new(value: f64, grad: f64) -> Unit {

           Unit {
            value: value,
            grad: grad,
           } 
        }
    }

    #[derive(Debug)]
    pub struct MultiplyGate {
        pub u0: Unit,
        pub u1: Unit,
        pub utop: Unit,
    }

    impl MultiplyGate {

        pub fn empty() -> MultiplyGate {
            MultiplyGate {
                u0: Unit::empty(),
                u1: Unit::empty(),
                utop: Unit::empty(),
            }
        }
        
        pub fn forward(mut self, u0: Unit, u1: Unit) -> MultiplyGate {
            
            self.u0 = u0;
            self.u1 = u1;
            self.utop = Unit::new(self.u0.value * self.u1.value, 0.0);
            self
        }

        pub fn backward(mut self) -> MultiplyGate {
            
            self.u0.grad = self.u0.grad + self.u1.value*self.utop.grad;
            self.u1.grad = self.u1.grad + self.u0.value*self.utop.grad;
            self
        }
    }
    
    #[derive(Debug)]
    pub struct AddGate {
        pub u0: Unit,
        pub u1: Unit,
        pub utop: Unit,
    }

    impl AddGate {
        
        pub fn empty() -> AddGate {
            AddGate {
                u0: Unit::empty(),
                u1: Unit::empty(),
                utop: Unit::empty(),
            }
        }

        pub fn forward(mut self, u0: Unit, u1: Unit) -> AddGate {
            
            self.u0 = u0;
            self.u1 = u1;
            self.utop = Unit::new(self.u0.value + self.u1.value, 0.0);
            self
        }

        pub fn backward(mut self) -> AddGate {
        
            self.u0.grad = self.u0.grad + 1.0 * self.utop.grad;
            self.u1.grad = self.u1.grad + 1.0 * self.utop.grad;
            self
        }
    }
    
    #[derive(Debug)]
    pub struct SigmoidGate {
        pub u0: Unit,
        pub utop: Unit,
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

    impl SigmoidGate {
       
        pub fn empty() -> 
            SigmoidGate 
            { 
                SigmoidGate {
                  u0: Unit::empty(),
                  utop: Unit::empty(),
                }
            }

        pub fn forward(mut self, u0: Unit) -> SigmoidGate {

           self.u0 = u0;
           self.utop = Unit::new(sig(self.u0.value), 0.0);
           self
        }

        pub fn backward(mut self) -> SigmoidGate {
        
           let s = sig(self.u0.value);
           self.u0.grad = self.u0.grad + (s * (1.0 - s)) * self.utop.grad;
           self
        }
    }
}

#[cfg(test)]
mod tests {
    

    use super::*;
    
    #[test]
    fn test_forward_neurons() {
        
        let mut a: foo::Unit = foo::Unit::new(1.0, 0.0);
        let mut b: foo::Unit = foo::Unit::new(2.0, 0.0);
        let mut c: foo::Unit = foo::Unit::new(-3.0, 0.0);
        let mut x: foo::Unit = foo::Unit::new(-1.0, 0.0);
        let mut y: foo::Unit = foo::Unit::new(3.0, 0.0);

        let mut mulg0: foo::MultiplyGate = foo::MultiplyGate::empty();
        let mut mulg1: foo::MultiplyGate = foo::MultiplyGate::empty();
        let mut addg0: foo::AddGate = foo::AddGate::empty();
        let mut addg1: foo::AddGate = foo::AddGate::empty();
        let mut sg0: foo::SigmoidGate = foo::SigmoidGate::empty();

            
        let mut ax: foo::MultiplyGate = foo::MultiplyGate::forward(mulg0, a, x);
        let mut by: foo::MultiplyGate = foo::MultiplyGate::forward(mulg1, b, y);
        let mut axpby: foo::AddGate = foo::AddGate::forward(addg0, ax.utop, by.utop);
        let mut axpbypc: foo::AddGate = foo::AddGate::forward(addg1, axpby.utop, c);
        let mut s: foo::SigmoidGate = foo::SigmoidGate::forward(sg0, axpbypc.utop);

        println!("AX => {:?}", ax.utop.value);
        println!("BY => {:?}", by.utop.value);
        println!("AXPBY => {:?}", axpby.utop.value);
        println!("AXPBYPC => {:?}", axpbypc.utop.value);
        println!("S => {:?}", s.utop.value);
    }

    #[test]
    fn it_works() {
        
        let mut sigmoid: foo::SigmoidGate = 
            foo::SigmoidGate {
                u0 : foo::Unit::new(1.0, 2.0),
                utop : foo::Unit::new(0.0, 0.0),
            };
        
        sigmoid = foo::SigmoidGate::forward(sigmoid, foo::Unit::new(-10.0, 2.0));
        sigmoid = foo::SigmoidGate::backward(sigmoid);
    }
}