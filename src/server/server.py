from utils import setup_japan_time_logging, get_logger, load_mnist
from model.net import Net
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flask import Flask, request, jsonify
import threading
import time
import sys
import os
# モデルディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

setup_japan_time_logging()
logger = get_logger(__name__)

app = Flask(__name__)

global_model = Net()
current_round = 0
total_rounds = 3
num_clients = 3
client_updates = {}
round_completion = threading.Event()

test_dataset = load_mnist(root='../../data', train=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


def list_to_tensor(lst):
    return torch.tensor(lst)


@app.route('/get_model', methods=['GET'])
def get_model():
    global current_round
    client_address = request.remote_addr
    logger.info(
        f"Sending global model to client {client_address} for round {current_round}")
    model_state = {k: tensor_to_list(v)
                   for k, v in global_model.state_dict().items()}
    return jsonify({"model_state": model_state, "round": current_round})


@app.route('/update_model', methods=['POST'])
def update_model():
    global global_model, current_round, client_updates, complete_all_rounds
    complete_all_rounds = False
    client_model = request.json['model_state']
    client_round = request.json['round']
    client_address = request.remote_addr

    if client_round != current_round:
        return jsonify({"status": "error", "message": "Client round does not match current round"}), 400

    logger.info(
        f"Received updated model from client {client_address} for round {current_round}")

    client_updates[client_address] = {
        k: list_to_tensor(v) for k, v in client_model.items()}

    if len(client_updates) == num_clients:
        perform_fedavg()
        current_round += 1
        client_updates.clear()
        if current_round < total_rounds:
            logger.info(f"Starting round {current_round}")
        else:
            complete_all_rounds = True
            logger.info("All rounds completed")
        round_completion.set()

    return jsonify({"status": "success", "round": current_round})


def evaluate_model(model, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def perform_fedavg():
    global global_model

    averaged_state_dict = {}
    for key in global_model.state_dict().keys():
        averaged_state_dict[key] = torch.stack(
            [updates[key] for updates in client_updates.values()]).mean(dim=0)

    global_model.load_state_dict(averaged_state_dict)
    logger.info(f"Global model updated using FedAvg for round {current_round}")

    # モデルの評価
    test_loss, accuracy = evaluate_model(global_model)
    logger.info(
        f"Round {current_round} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


@app.route('/wait_for_round_completion', methods=['GET'])
def wait_for_round_completion():
    global round_completion
    round_completion.wait()
    time.sleep(2)  # Wait for other clients
    round_completion.clear()
    return jsonify({"status": "success", "round": current_round, "finish": complete_all_rounds})


def start_server():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    initial_loss, initial_accuracy = evaluate_model(global_model)
    logger.info(
        f"Initial model - Test Loss: {initial_loss:.4f}, Accuracy: {initial_accuracy:.2f}%")
    start_server()
