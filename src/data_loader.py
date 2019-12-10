import numpy as np
import os


def load_data(args):
    n_user, n_item, ori_train_data, ori_eval_data, ori_test_data, rating = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)

    # enrich rs train, eval, test. Add one more "null relation" as last index
    train_data = enrich_rs_train(ori_train_data, kg, n_relation)
    eval_data = np.c_[ori_eval_data, np.ones(ori_eval_data.shape[0]) * n_relation]
    test_data = np.c_[ori_test_data, np.ones(ori_test_data.shape[0]) * n_relation]
    n_relation += 1
    # enrich kg. Add one more "null user" as last index. only use rs_train
    kg = enrich_kg(kg, ori_train_data, n_user)
    n_user += 1

    print('data loaded.')
    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, kg


def enrich_rs_train(data, kg, null_relation_id):
    # kg dict := head -> list(relations)
    kg_dict = {}
    for head, relation, tail in kg:
        if head not in kg_dict:
            kg_dict[head] = []
        kg_dict[head].append(relation)
    # enrich
    # sample relation for rs (user, item, rating) => (user, item, rating, relation_taking_item_as_head)
    new_data = []
    for user, item, rating in data:
        if item in kg_dict:
            for relation in kg_dict[item]:
                new_data.append([user, item, rating, relation])
        # for all pairs, we add null relation => easy for evaluation
        new_data.append([user, item, rating, null_relation_id])
    new_data = np.asarray(new_data, dtype=np.int32)
    return new_data


def enrich_kg(kg, rating, null_user_id):
    # rs_dict :=  item -> list(users)
    rs_dict = {}
    for user, item, _ in rating:
        if item not in rs_dict:
            rs_dict[item] = list()
        rs_dict[item].append(user)
    # enrich
    # sample user for kge (head, relation, tail) => (head, relation, tail, user_related_with_head)
    new_kg = []
    for head, relation, tail in kg:
        if head in rs_dict:
            for user in rs_dict[head]:
                new_kg.append([head, relation, tail, user])
        # for all triples, we add null user
        new_kg.append([head, relation, tail, null_user_id])
    new_kg = np.asarray(new_kg, dtype=np.int32)
    return new_kg


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np)


    return n_user, n_item, train_data, eval_data, test_data, rating_np


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg)

    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    return n_entity, n_relation, kg
