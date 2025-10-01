import random

def get_train_validation_test_participants(participants, number_of_test_participants, k, fold_number, random_seed):
    rng = random.Random(random_seed)
    participants = list(participants)
    rng.shuffle(participants)              

    test_participants = participants[:number_of_test_participants]
    train_validation_participants = participants[number_of_test_participants:]

    fold_size = len(train_validation_participants) // k

    if fold_size * k != len(train_validation_participants):
        raise ValueError("The number of participants is not evenly divisible by the number of folds.")
    
    start_index = fold_number * fold_size
    end_index = start_index + fold_size
    validation_participants = train_validation_participants[start_index:end_index]
    train_participants = train_validation_participants[:start_index] + train_validation_participants[end_index:]

    return train_participants, validation_participants, test_participants