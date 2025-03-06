import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
import yt_dlp
import io
import argparse
import subprocess
import numpy as np
from encodec import EncodecModel
from diffusers import DDPMScheduler


class AudioConvNet(nn.Module):
    def __init__(self, input_channels=32, output_channels=1024):
        super(AudioConvNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, output_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Initialize fc as None, to be computed later
        self.fc = None

    def _initialize_fc_layer(self, dummy_input, output_channels):
        # Pass the dummy input through the network to calculate the output size
        dummy_output = self._forward_convolutions(dummy_input)
        flat_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        self.fc = nn.Linear(flat_size, 1)  # Initialize the fully connected layer with correct size

    def _forward_convolutions(self, x):
        """Pass through the convolutional layers only, to compute output size."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        return x

    def forward(self, x):
        # Initialize the fully connected layer if it hasn't been initialized yet
        if self.fc is None:
            self._initialize_fc_layer(x, 1024)

        # Pass through the convolutional layers
        x = self._forward_convolutions(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # Pass through the fully connected layer
        return x


def get_trending_music():
    """Fetch URLs of trending music videos from YouTube, excluding premieres."""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'default_search': 'ytsearch',
        'force_generic_extractor': True,
        'playlistend': 10  # Get top 10 trending videos
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info("https://www.youtube.com/feed/trending?bp=4gINGgt5dG1hX2NoYXJ0cw%3D%3D", download=False)
        
        music_urls = []
        for entry in result.get('entries', []):
            if 'url' in entry:
                if entry.get("availability") == "premieres":
                    print(f"Skipping premiere: {entry.get('title', 'Unknown Title')}")
                    continue
                music_urls.append(entry['url'])
        
        return music_urls
    
    except yt_dlp.utils.ExtractorError as e:
        print(f"Error extracting YouTube trending music: {e}")
        return []


def stream_audio(youtube_url):
    """Stream YouTube audio and return waveform."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        audio_url = info['url']
    
    command = [
        "ffmpeg", "-i", audio_url, "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000", "pipe:1"
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio_data = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.tensor(audio_data).unsqueeze(0)  # Convert to torch tensor
    return waveform


def encode_audio(waveform):
    """Convert audio waveform to Encodec latent representation."""
    model = EncodecModel.encodec_model_24khz()
    model.eval()
    with torch.no_grad():
        encoded_frames = model.encode(waveform.unsqueeze(0))
    latent_audio = encoded_frames[0][0]
    latent_audio = latent_audio.float()
    return latent_audio


def train_step(latent_audio, model, optimizer):
    """Train a custom model on audio latent representations."""
    noise = torch.randn_like(latent_audio, dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(model(latent_audio), noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    """Main function to train custom model on YouTube trending music."""
    trending_urls = get_trending_music()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    for url in trending_urls:
        print(f"Processing: {url}")
        waveform = stream_audio(url)
        latent_audio = encode_audio(waveform)

        model = AudioConvNet(input_channels=latent_audio.shape[1], output_channels=1024)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        print("Latent Audio Shape:", latent_audio.shape)

        for epoch in range(5):  # Train on each audio sample for 5 epochs
            loss = train_step(latent_audio, model, optimizer)
            print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    main()
