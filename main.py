from src.model import build_model
from src.preprocessing import load_data
from src.predict import predict_image
import matplotlib.pyplot as plt

# Load data
train, val = load_data()

# Build model
model = build_model()

# Train model
history = model.fit(train, validation_data=val, epochs=5)

# Evaluate
loss, acc = model.evaluate(val)
print("Accuracy:", acc)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.savefig("outputs/accuracy.png")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.savefig("outputs/loss.png")
plt.show()



from src.predict import predict_image

result = predict_image(model, "data/normal/000001.jpg")
print("Prediction:", result)