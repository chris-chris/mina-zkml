pub mod operations;
pub mod proof;
pub mod wiring;

use mina_curves::pasta::Vesta;
use poly_commitment::ipa::OpeningProof;

pub type ZkOpeningProof = OpeningProof<Vesta>;

pub use wiring::ModelCircuitBuilder;
