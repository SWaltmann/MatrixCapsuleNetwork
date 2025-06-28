import json
import matplotlib.pyplot as plt

run_a = input("Run A: ")
run_b = input("Run B: ")
with open(f"training_history{run_a}.json", "r") as f:
    loaded_history = json.load(f)

with open(f"test_history{run_a}.json", "r") as f:
    test_history = json.load(f)

with open(f"training_history{run_b}.json", "r") as f:
    loaded_history2 = json.load(f)

with open(f"test_history{run_b}.json", "r") as f:
    test_history2 = json.load(f)


loss = loaded_history['loss']
acc = loaded_history['categorical_accuracy']

test_acc = test_history['accuracy']
print(f"Min training loss: {min(loss):.4f}")
print(f"Max accuracy: {max(acc):.4f}")
print(f"Max test accuracy: {max(test_acc):.4f}")



plt.plot(test_acc, label='Test Accuracy')
plt.plot(loaded_history['categorical_accuracy'], label='Training Accuracy')
plt.plot(loaded_history['loss'], label='Loss')
plt.title(run_a)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(loaded_history['loss'], label=f'{run_a}')
plt.plot(loaded_history2['loss'], label=f'{run_b}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(test_history['accuracy'], label=f'{run_a}')
plt.plot(test_history2['accuracy'], label=f'{run_b}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()