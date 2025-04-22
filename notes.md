# Multiprocessing не работает

```
[2024-02-01 23:03:40,535: ERROR/ForkPoolWorker-6] Task ml_training.base_train.tasks.train_ml_model[b4e5c9ba-de5a-43ea-b852-2c2f405c8e8f] raised unexpected: AssertionError('daemonic processes are not allowed to have children')
Traceback (most recent call last):
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/celery/app/trace.py", line 477, in trace_task
    R = retval = fun(*args, **kwargs)
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/celery/app/trace.py", line 760, in __protected_call__
    return self.run(*args, **kwargs)
  File "/home/student2/workspace/ml_traning/ml_training/base_train/tasks.py", line 33, in train_ml_model
    result = yolo_train(creds)
  File "/home/student2/workspace/ml_traning/ml_training/base_train/train.py", line 176, in yolo_train
    wrapper.train()
  File "/home/student2/workspace/ml_traning/ml_training/wrappers/yolo/wrapper.py", line 28, in train
    self.model.train(**self.config)
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/engine/model.py", line 391, in train
    self.trainer.train()
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/engine/trainer.py", line 208, in train
    self._do_train(world_size)
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/engine/trainer.py", line 322, in _do_train
    self._setup_train(world_size)
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/engine/trainer.py", line 286, in _setup_train
    self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/models/yolo/detect/train.py", line 55, in get_dataloader
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/data/build.py", line 114, in build_dataloader
    return InfiniteDataLoader(
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/ultralytics/data/build.py", line 40, in __init__
    self.iterator = super().__iter__()
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 438, in __iter__
    return self._get_iterator()
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 386, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/home/student2/workspace/ml_traning/venv/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1039, in __init__
    w.start()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 118, in start
    assert not _current_process._config.get('daemon'), \
AssertionError: daemonic processes are not allowed to have children
```

# Воркеры держат память GPU

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       822      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A    591579      C   ...D-PC18:ForkPoolWorker-13]      790MiB |
+-----------------------------------------------------------------------------+

