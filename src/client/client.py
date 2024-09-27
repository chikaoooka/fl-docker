from utils import setup_japan_time_logging, get_logger, load_mnist, create_non_iid_data
from model.net import Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import requests
import sys
import time
import os
# モデルディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

setup_japan_time_logging()
logger = get_logger(__name__)


def client_update(client_model, optimizer, train_loader, epochs, client_id):
    client_model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        average_loss = total_loss / len(train_loader)
        logger.info(
            f"Client {client_id + 1}, Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")


def run_client(client_id):
    logger.info(f"Client {client_id + 1} starting")

    dataset = load_mnist(root='../../data')
    client_indices = create_non_iid_data(dataset, 3, client_id)
    client_data = Subset(dataset, client_indices)
    client_loader = DataLoader(client_data, batch_size=32, shuffle=True)

    client_model = Net()
    optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.5)

    current_round = -1
    while True:
        response = requests.get('http://server:5000/get_model')
        data = response.json()
        global_model_state = data['model_state']
        server_round = data['round']

        if server_round > current_round:
            current_round = server_round
            logger.info(
                f"Client {client_id + 1} starting round {current_round}")
            client_model.load_state_dict(
                {k: torch.tensor(v) for k, v in global_model_state.items()})
            logger.info(
                f"Client {client_id + 1} received global model for round {current_round}")

            client_update(client_model, optimizer, client_loader,
                          epochs=2, client_id=client_id)
            logger.info(
                f"Client {client_id + 1} completed local training for round {current_round}")

            updated_model_state = {k: v.cpu().numpy().tolist()
                                   for k, v in client_model.state_dict().items()}
            response = requests.post('http://server:5000/update_model',
                                     json={"model_state": updated_model_state, "round": current_round})

            if response.status_code == 200:
                logger.info(
                    f"Client {client_id + 1} successfully sent updated model to server for round {current_round}")
            else:
                logger.error(
                    f"Client {client_id + 1} failed to update the global model on the server for round {current_round}")

            # Wait for all clients to complete the round
            response = requests.get(
                'http://server:5000/wait_for_round_completion')
            if response.status_code == 200:
                data = response.json()
                if data["finish"] == True:
                    logger.info(
                        f"Client {client_id + 1} completed all rounds. Exiting.")
                    break
                else:
                    logger.info(
                        f"Client {client_id + 1} completed round {current_round}")
            else:
                logger.error(
                    f"Client {client_id + 1} failed to synchronize after round {current_round}")

        elif server_round < current_round:
            logger.info(
                f"Client {client_id + 1} completed all rounds. Exiting.")
            break

        time.sleep(5)  # Wait before checking for the next round


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1]) - 1  # 0-indexed
    run_client(client_id)
