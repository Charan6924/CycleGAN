import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("loss_plots/cyclegan_losses.csv")

# Plot
plt.figure(figsize=(10,6))

plt.plot(df.index, df["Generator Loss"], label="Generator Loss", alpha=0.8)
plt.plot(df.index, df["Discriminator Loss"], label="Discriminator Loss", alpha=0.8)

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.grid(True)
plt.show()
