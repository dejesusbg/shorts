# AI Shorts Generator

This project is a fully automated video generator that uses the Gemini AI API to create stunning videos from a simple JSON input. It leverages Gemini for generating images, optimizing voiceover scripts, and even creating background music. The entire process is containerized with Docker, making it easy to run and deploy.

## Features

- **AI-Powered Image Generation:** Creates images from text prompts using Gemini's Imagen 3 model.
- **Natural Voiceover:** Enhances the voiceover script with natural pauses and emphasis for a more engaging narration.
- **Procedural Background Music:** Generates background music that matches the mood of the video.
- **Subtitle Integration:** Automatically adds subtitles to the video based on the provided SRT content.
- **Dockerized Environment:** Comes with a `Dockerfile` and `docker-compose.yml` for easy setup and execution.
- **Customizable Output:** Allows for custom output names and provides a structured output directory.

## Getting Started

### Prerequisites

- Python 3.11 or later
- Docker (for containerized execution)
- A Gemini API key. You can get one from the [Google AI Studio](https://makersuite.google.com/app/apikey).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dejesusbg/shorts.git
    cd shorts
    ```

2.  **Set up the Gemini API Key:**
    ```bash
    export GEMINI_API_KEY='your-key'
    ```

3.  **Install dependencies (for local execution):**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Local Execution

1.  **Create an input JSON file.** You can use the provided example or create your own. The script can also generate an example for you.
    ```bash
    python video_generator.py
    ```
    This will create an `example_input.json` file.

2.  **Run the video generator:**
    ```bash
    python video_generator.py example_input.json my_awesome_video
    ```
    The output video will be saved in the `output` directory.

### Docker Execution

1.  **Build the Docker image:**
    ```bash
    docker-compose build
    ```

2.  **Run the video generator using Docker Compose:**
    ```bash
    docker-compose up
    ```
    This will run the `video-generator` service defined in the `docker-compose.yml` file, which uses the `example_input.json` to generate a video named `my_video.mp4`.

### Input JSON Format

The input JSON file has the following structure:

```json
{
  "srt_content": "SRT content for subtitles",
  "image_prompts": "Prompts for image generation",
  "vo_notes": {
    "speed": 145,
    "emotion": "intimate and tense, building suspense"
  },
  "metadata": {
    "title": "Storm Night",
    "description": "A short story about a storm."
  }
}
```

-   `srt_content`: A string containing the subtitles in SRT format.
-   `image_prompts`: A string containing one or more image prompts. Each prompt should be enclosed in `[IMAGE]` tags and include a timestamp.
-   `vo_notes`: An object containing notes for the voiceover, such as speed and emotion.
-   `metadata`: An object containing metadata for the video, such as the title and description.

## Docker Integration

The `docker-compose.yml` file defines two services:

-   `video-generator`: The main service for generating videos.
-   `video-generator-dev`: A development service that provides an interactive shell for debugging and development.

To run the development service, use the following command:
```bash
docker-compose run --rm video-generator-dev
```

This will start a bash session inside the container, allowing you to run the script manually and access the files.
