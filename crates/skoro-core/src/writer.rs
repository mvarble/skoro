use std::sync::{Arc, Mutex, PoisonError};

use crate::Failure;
use crate::model::{Model, Snapshot};
use crate::scope::{self, Draft, Frame};

/// Publishes snapshots of one branch of a model, each built in a scope.
#[derive(Debug)]
pub struct Writer {
    latest: Snapshot,
    cell: Arc<Mutex<Snapshot>>,
}

/// Loads the newest snapshot a writer published, from any thread.
#[derive(Clone, Debug)]
pub struct Reader {
    cell: Arc<Mutex<Snapshot>>,
}

impl Writer {
    /// A writer of an empty model.
    pub fn new() -> Writer {
        Writer::from_snapshot(Snapshot::new(Model::new()))
    }

    pub(crate) fn from_snapshot(snapshot: Snapshot) -> Writer {
        Writer {
            cell: Arc::new(Mutex::new(snapshot.clone())),
            latest: snapshot,
        }
    }

    /// Runs `body` with a building scope open on this thread, and publishes what it built. A
    /// scope that recorded a failure publishes nothing and returns the failure; one that panics
    /// drops its draft as it unwinds.
    pub fn build<R>(&mut self, body: impl FnOnce() -> R) -> Result<R, Failure> {
        let draft = Draft::new(self.latest.0.as_ref().clone());
        let (result, frame) = scope::open(Frame::Scope(draft), body);
        let Frame::Scope(draft) = frame else {
            unreachable!("build gets back the scope it opened")
        };
        if let Some(failure) = draft.failure {
            return Err(failure);
        }
        self.publish(draft.model);
        Ok(result)
    }

    /// Publishes `model` as the branch's newest snapshot.
    pub(crate) fn publish(&mut self, model: Model) {
        self.latest = Snapshot::new(model);
        // The lock is held for one assignment, so a reader loads the model before the change or
        // after it.
        *self.cell.lock().unwrap_or_else(PoisonError::into_inner) = self.latest.clone();
    }

    pub fn snapshot(&self) -> Snapshot {
        self.latest.clone()
    }

    pub fn reader(&self) -> Reader {
        Reader {
            cell: self.cell.clone(),
        }
    }
}

impl Default for Writer {
    fn default() -> Writer {
        Writer::new()
    }
}

impl Reader {
    /// The newest snapshot the writer published.
    pub fn latest(&self) -> Snapshot {
        self.cell
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}
