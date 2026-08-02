pub trait SequenceMutator: 'static + Send + Sync + std::fmt::Debug {
    fn input_indices(&self) -> &[usize];
    fn output_index(&self) -> usize;
}
