from datasets.dataset_cifar import load_cifar10
from datasets.dataset_stanford_dogs import load_stanford_dogs

def get_data(
    dataset_configs,
    transform_train,
    transform_test,
    seed_worker_fn,
    seed_generator,
    logger,
    test_batch_size=None,
    debug=False
):
    if dataset_configs["type"] == "cifar10":
        train_loader, val_loader, test_loader = load_cifar10(
            dataset_configs,
            transform_train,
            transform_test,
            seed_worker_fn,
            seed_generator,
            logger,
            test_batch_size=test_batch_size,
            debug=debug
        )
    elif dataset_configs["type"] == "stanford_dogs":
        train_loader, val_loader, test_loader = load_stanford_dogs(
            dataset_configs,
            transform_train,
            transform_test,
            seed_worker_fn,
            seed_generator,
            logger,
            test_batch_size=test_batch_size,
            debug=debug
        )
    
    return train_loader, val_loader, test_loader
