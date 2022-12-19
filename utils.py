import torch
from neo4j import GraphDatabase
from neo4_connection import USER, PWD, URL


class MovieDataset:
    def __init__(self, users, movies, ratings) -> None:
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, item) -> dict:

        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor(users), #dtype=torch.long
            "movies": torch.tensor(movies), #dtype=torch.long
            "ratings": torch.tensor(ratings) #dtype=torch.long
        }


def preprocess_neo4j_db(file_path:str) -> None:
    """Execute some type casts and data preprocessing steps on neo4j."""

    with open(file_path, mode='r', encoding='utf-8') as f:
        query = f.read()

    queries = [query for query in query.split(';')[:-1]] 

    driver = GraphDatabase.driver(uri=URL, auth=(USER, PWD))
    with driver.session() as session:
        for query in queries:
            result = session.run(query)
            print(result)


def interaction_matrix_to_adj_matrix(int_mat:torch.Tensor) -> torch.Tensor:
    """ Converts a interaction matrix to an axjecency matrix.
    
        In bipartite graph, interaction matrix is not the same as adjacency matrix, 
        because adjacency matrix expecteds row index and col index to refer to the 
        same node perform the conversion between interaction matrix (r_mat) and 
        adjacency matrix (adj_mat)
            ( 0    R )
        A = ( R_T  0 )

        so if dimension of R is  n_user x n_item
        then dimension of A is (n_user+n_item) x (n_user+n_item)
    """
    adj_size = int_mat.size()[0] + int_mat.size()[1]

    z_mat = torch.zeros((adj_size - int_mat.size()[1], adj_size - int_mat.size()[1]))
    a = torch.cat([z_mat, int_mat], dim=1)
    
    z_mat = torch.zeros((adj_size - int_mat.T.size()[1], adj_size - int_mat.T.size()[1]))
    b = torch.cat([int_mat.T, z_mat], dim=1)

    adj_matrix = torch.cat([a, b], dim=0)
    return adj_matrix


def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()