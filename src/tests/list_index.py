import joblib

def list_available_ids(filename="C:\\Users\\allan\\Documents\\recommendation_system\\src\\models\\model.pkl"):
    """
    Lista os IDs disponíveis no índice do modelo.

    Args:
        filename (str): Nome do arquivo do modelo consolidado.

    Returns:
        list: Lista de IDs disponíveis.
    """
    model = joblib.load(filename)
    index = model["index"]
    print(list(index))


list_available_ids()
