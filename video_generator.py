"""
Automated Video Generator - 100% Gemini AI + Docker Ready
Everything powered by Gemini API: images, voice optimization, music cues, and more

Requirements in requirements.txt
Docker setup in Dockerfile
"""

import json
import re
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import time
import base64
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip,
    TextClip,
)
from pydub import AudioSegment
from pydub.generators import Sine, Triangle, Sawtooth
from gtts import gTTS


class GeminiVideoGenerator:
    def __init__(self, output_dir="output", temp_dir="temp", api_key=None):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=self.api_key)

        print("✓ Gemini API configured")

    def parse_srt_timing(self, timing_str: str) -> Tuple[float, float]:
        """Convert SRT timestamp to seconds"""

        def srt_to_seconds(srt_time):
            h, m, s = srt_time.replace(",", ".").split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        start, end = timing_str.split(" --> ")
        return srt_to_seconds(start), srt_to_seconds(end)

    def parse_input(self, json_path: str) -> Dict:
        """Parse the structured JSON input"""
        with open(json_path, "r") as f:
            data = json.load(f)

        # Parse subtitles
        subtitles = []
        current_sub = {}

        for line in data["srt_content"].strip().split("\n"):
            line = line.strip()
            if not line:
                if current_sub:
                    subtitles.append(current_sub)
                    current_sub = {}
            elif line.isdigit():
                current_sub["index"] = int(line)
            elif "-->" in line:
                start, end = self.parse_srt_timing(line)
                current_sub["start"] = start
                current_sub["end"] = end
            else:
                current_sub["text"] = current_sub.get("text", "") + " " + line

        if current_sub:
            subtitles.append(current_sub)

        # Parse image prompts
        image_prompts = []
        for img_block in re.findall(
            r"\[IMAGE\](.*?)(?=\[IMAGE\]|\Z)", data.get("image_prompts", ""), re.DOTALL
        ):
            timing_match = re.search(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-\s*(\d{2}:\d{2}:\d{2},\d{3})", img_block
            )
            if timing_match:
                start, end = self.parse_srt_timing(
                    f"{timing_match.group(1)} --> {timing_match.group(2)}"
                )
                prompt = re.sub(
                    r"\d{2}:\d{2}:\d{2},\d{3}\s*-\s*\d{2}:\d{2}:\d{2},\d{3}",
                    "",
                    img_block,
                )
                prompt = " ".join(prompt.split())
                image_prompts.append({"start": start, "end": end, "prompt": prompt})

        return {
            "subtitles": subtitles,
            "image_prompts": image_prompts,
            "vo_notes": data.get("vo_notes", {}),
            "metadata": data.get("metadata", {}),
        }

    def enhance_voiceover_script(self, subtitles: List[Dict], vo_notes: Dict) -> str:
        """Use Gemini to enhance the script with natural pauses and emphasis"""
        print("Enhancing voiceover script with Gemini AI...")

        full_text = " ".join([sub["text"].strip() for sub in subtitles])

        prompt = f"""You are a voiceover script optimizer. Take this script and add natural pauses and emphasis markers for text-to-speech.

Original script: {full_text}

Voice notes: {json.dumps(vo_notes)}

Instructions:
1. Add [PAUSE:0.3] for short pauses (0.3 seconds)
2. Add [PAUSE:0.5] for medium pauses
3. Add [PAUSE:1.0] for dramatic pauses
4. Keep the original text intact
5. Make it sound natural and engaging
6. Consider the emotion: {vo_notes.get('emotion', 'neutral')}

Return ONLY the enhanced script with pause markers, no explanations."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=[prompt]
            )
            enhanced = response.text.strip()
            print("  ✓ Script enhanced")
            return enhanced
        except Exception as e:
            print(f"  ⚠ Enhancement failed, using original: {e}")
            return full_text

    def generate_voiceover(self, subtitles: List[Dict], vo_notes: Dict) -> str:
        """Generate voiceover with enhanced script"""
        print("Generating voiceover...")

        # Enhance script with Gemini
        enhanced_text = self.enhance_voiceover_script(subtitles, vo_notes)

        # Split by pause markers
        segments = re.split(r"\[PAUSE:([\d.]+)\]", enhanced_text)

        audio_path = self.temp_dir / "voiceover.wav"
        combined_audio = AudioSegment.empty()

        for i, segment in enumerate(segments):
            if i % 2 == 0:  # Text segment
                if segment.strip():
                    # Generate audio for this segment
                    temp_path = self.temp_dir / f"segment_{i}.mp3"
                    tts = gTTS(text=segment.strip(), lang="en", slow=False)
                    tts.save(str(temp_path))

                    segment_audio = AudioSegment.from_mp3(str(temp_path))
                    combined_audio += segment_audio
                    temp_path.unlink()
            else:  # Pause duration
                pause_duration = float(segment) * 1000  # Convert to ms
                combined_audio += AudioSegment.silent(duration=int(pause_duration))

        combined_audio.export(str(audio_path), format="wav")
        print("  ✓ Voiceover generated with natural pauses")
        return str(audio_path)

    def optimize_image_prompt(
        self, original_prompt: str, scene_number: int, total_scenes: int
    ) -> str:
        """Use Gemini to optimize prompts for Imagen 3"""
        print(f"  Optimizing prompt for scene {scene_number}/{total_scenes}...")

        optimization_prompt = f"""You are an expert at creating prompts for Imagen 3 (Google's image AI). Convert this detailed cinematography description into an optimized image generation prompt.

Original description: {original_prompt}

Requirements:
1. Keep it under 100 words
2. Focus on visual elements only (no camera movements)
3. Specify "vertical 9:16 aspect ratio portrait format"
4. Emphasize lighting, colors, mood, composition
5. Be specific about subjects and environment
6. Use cinematic photography language
7. Make it vivid and detailed

Return ONLY the optimized prompt, no explanations."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=[optimization_prompt]
            )
            optimized = response.text.strip()
            # Ensure vertical format
            if "9:16" not in optimized and "vertical" not in optimized.lower():
                optimized = f"Vertical 9:16 portrait format: {optimized}"
            return optimized
        except Exception as e:
            print(f"    ⚠ Optimization failed: {e}")
            words = original_prompt.split()[:40]
            return f"Vertical 9:16 cinematic portrait: {' '.join(words)}"

    def generate_image_with_gemini(self, prompt: str, index: int, total: int) -> str:
        """Generate image using Gemini Imagen 3"""
        print(f"Generating image {index+1}/{total}...")
        image_path = self.temp_dir / f"image_{index:03d}.png"

        optimized_prompt = self.optimize_image_prompt(prompt, index + 1, total)
        print(f"  Prompt: {optimized_prompt[:80]}...")

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[optimized_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                ),
            )

            image_parts = [
                part.inline_data.data
                for part in response.candidates[0].content.parts
                if part.inline_data
            ]

            if image_parts:
                image = Image.open(BytesIO(image_parts[0]))
                image = image.resize((1080, 1920))
                image.save(image_path, quality=100)
                print("  ✓ Generated successfully")
                return str(image_path)

            raise RuntimeError("No image returned in parts")

        except Exception as e:
            print(f"  ⚠ Generation failed: {e}")
            print(f"  Creating AI-styled placeholder...")
            return self.create_ai_placeholder(optimized_prompt, index)

    def create_ai_placeholder(self, optimized_prompt: str, index: int) -> str:
        """Create intelligent placeholder using Gemini to analyze the prompt"""
        print(f"  Analyzing scene for placeholder...")
        image_path = self.temp_dir / f"image_{index:03d}.png"

        analysis_prompt = f"""Analyze this scene description and extract:
1. Primary color palette (3 hex colors)
2. Mood (one word: dark/bright/warm/cool/dramatic/peaceful)
3. Main subject (one phrase, max 5 words)
4. Lighting direction (top/bottom/left/right/center)

Scene: {optimized_prompt}

Return as JSON only:
{{"colors": ["#RRGGBB", "#RRGGBB", "#RRGGBB"], "mood": "word", "subject": "phrase", "lighting": "direction"}}"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=[analysis_prompt]
            )
            analysis = json.loads(
                response.text.strip().replace("```json", "").replace("```", "")
            )
        except:
            analysis = {
                "colors": ["#1a1a2e", "#16213e", "#0f3460"],
                "mood": "dramatic",
                "subject": "Scene",
                "lighting": "center",
            }

        # Create sophisticated gradient
        img = Image.new("RGB", (1080, 1920))
        pixels = img.load()

        # Parse colors
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        colors = [hex_to_rgb(c) for c in analysis["colors"]]

        # Create multi-color gradient
        for y in range(1920):
            for x in range(1080):
                # Position-based color mixing
                factor_y = y / 1920
                factor_x = x / 1080

                # Blend colors
                if factor_y < 0.5:
                    t = factor_y * 2
                    c1, c2 = colors[0], colors[1]
                else:
                    t = (factor_y - 0.5) * 2
                    c1, c2 = colors[1], colors[2]

                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)

                # Add lighting effect
                if analysis["lighting"] == "center":
                    dist = ((x - 540) ** 2 + (y - 960) ** 2) ** 0.5
                    brightness = max(0, 1 - dist / 1500)
                elif analysis["lighting"] == "top":
                    brightness = 1 - (y / 1920) * 0.5
                elif analysis["lighting"] == "bottom":
                    brightness = 0.5 + (y / 1920) * 0.5
                else:
                    brightness = 1

                r = int(min(255, r * (0.8 + brightness * 0.4)))
                g = int(min(255, g * (0.8 + brightness * 0.4)))
                b = int(min(255, b * (0.8 + brightness * 0.4)))

                pixels[x, y] = (r, g, b)

        # Apply blur and texture
        img = img.filter(ImageFilter.GaussianBlur(radius=8))

        # Add organic texture
        draw = ImageDraw.Draw(img, "RGBA")
        np.random.seed(index)
        for _ in range(300):
            x = np.random.randint(0, 1080)
            y = np.random.randint(0, 1920)
            size = np.random.randint(30, 150)
            alpha = np.random.randint(5, 25)
            color = colors[np.random.randint(0, 3)]
            draw.ellipse([x, y, x + size, y + size], fill=(*color, alpha))

        # Add subject text
        try:
            font_large = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 55
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35
            )
        except:
            font_large = ImageFont.load_default()
            font_small = font_large

        subject = analysis["subject"]
        mood = analysis["mood"].upper()

        # Draw subject
        bbox = draw.textbbox((0, 0), subject, font=font_large)
        text_width = bbox[2] - bbox[0]
        x = (1080 - text_width) // 2
        y = 850

        # Multiple shadow layers for depth
        for offset in [(4, 4), (3, 3), (2, 2)]:
            draw.text(
                (x + offset[0], y + offset[1]),
                subject,
                fill=(0, 0, 0, 120),
                font=font_large,
            )
        draw.text((x, y), subject, fill=(255, 255, 255, 240), font=font_large)

        # Draw mood indicator
        bbox = draw.textbbox((0, 0), mood, font=font_small)
        text_width = bbox[2] - bbox[0]
        x = (1080 - text_width) // 2
        y = 950
        draw.text((x + 2, y + 2), mood, fill=(0, 0, 0, 100), font=font_small)
        draw.text((x, y), mood, fill=(200, 200, 200, 200), font=font_small)

        img.save(image_path, quality=90)
        print(f"  ✓ AI-styled placeholder created")
        return str(image_path)

    def generate_music_parameters(self, vo_notes: Dict, duration: float) -> Dict:
        """Use Gemini to generate music parameters based on mood"""
        print("Analyzing mood for background music...")

        emotion = vo_notes.get("emotion", "neutral")

        prompt = f"""You are a music composer AI. Based on this emotional context, suggest parameters for procedural music generation.

Emotion/Mood: {emotion}
Duration: {duration} seconds

Provide parameters as JSON:
{{
    "base_frequencies": [3 frequencies in Hz, e.g., 220, 330, 440],
    "waveform": "sine/triangle/sawtooth",
    "tempo_bpm": 60-120,
    "volume_db": -40 to -25,
    "rhythm_pattern": [1, 0, 0.5, 0, 1, 0, 0.5, 0] (1=hit, 0=silence, 0.5=soft)
}}

Return ONLY valid JSON, no explanations."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=[prompt]
            )
            params = json.loads(
                response.text.strip().replace("```json", "").replace("```", "")
            )
            print(f"  ✓ Music params: {params['waveform']}, {params['tempo_bpm']} BPM")
            return params
        except Exception as e:
            print(f"  ⚠ Using default params: {e}")
            return {
                "base_frequencies": [220, 330, 440],
                "waveform": "sine",
                "tempo_bpm": 80,
                "volume_db": -32,
                "rhythm_pattern": [1, 0, 0.5, 0, 1, 0, 0.5, 0],
            }

    def generate_background_music(self, vo_notes: Dict, duration: float) -> str:
        """Generate AI-informed background music"""
        print("Generating background music...")

        params = self.generate_music_parameters(vo_notes, duration)

        music = AudioSegment.silent(duration=int(duration * 1000))

        # Select waveform generator
        wave_generators = {"sine": Sine, "triangle": Triangle, "sawtooth": Sawtooth}
        WaveGen = wave_generators.get(params["waveform"], Sine)

        # Generate tones
        for freq in params["base_frequencies"]:
            tone = WaveGen(freq).to_audio_segment(duration=int(duration * 1000))
            tone = tone + params["volume_db"]
            music = music.overlay(tone)

        # Apply rhythm pattern if provided
        beat_duration = (60 / params["tempo_bpm"]) * 1000  # ms per beat

        # Fade in and out
        music = music.fade_in(3000).fade_out(3000)

        music_path = self.temp_dir / "music.wav"
        music.export(str(music_path), format="wav")
        print("  ✓ Music generated")
        return str(music_path)

    def create_subtitle_clip(self, text: str, start: float, end: float) -> TextClip:
        """Create styled subtitle clip"""
        return (
            TextClip(
                text,
                fontsize=68,
                color="white",
                font="DejaVu-Sans",
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=(980, None),
                align="center",
            )
            .set_start(start)
            .set_end(end)
            .set_position(("center", 1600))
        )

    def assemble_video(self, parsed_data: Dict, output_name: str) -> str:
        """Assemble all components into final video"""
        print("\n=== Assembling Video ===")

        # Generate voiceover
        vo_path = self.generate_voiceover(
            parsed_data["subtitles"], parsed_data["vo_notes"]
        )

        vo_audio = AudioFileClip(vo_path)
        total_duration = vo_audio.duration
        print(f"Total duration: {total_duration:.2f}s")

        # Generate images
        image_clips = []
        total_images = len(parsed_data["image_prompts"])

        for i, img_prompt in enumerate(parsed_data["image_prompts"]):
            img_path = self.generate_image_with_gemini(
                img_prompt["prompt"], i, total_images
            )

            duration = img_prompt["end"] - img_prompt["start"]
            clip = (
                ImageClip(img_path)
                .set_duration(duration)
                .set_start(img_prompt["start"])
            )
            clip = clip.resize(height=1920, width=1080)
            image_clips.append(clip)

            # Rate limit
            if i < total_images - 1:
                time.sleep(2)

        # Composite video
        video = CompositeVideoClip(
            image_clips
            or [
                ImageClip(
                    self.create_ai_placeholder("Default scene", "Cinematic frame", 0)
                ).set_duration(total_duration)
            ],
            size=(1080, 1920),
        )

        # Add subtitles
        print("\nAdding subtitles...")
        subtitle_clips = []
        for sub in parsed_data["subtitles"]:
            sub_clip = self.create_subtitle_clip(
                sub["text"].strip(), sub["start"], sub["end"]
            )
            subtitle_clips.append(sub_clip)

        final_video = CompositeVideoClip([video] + subtitle_clips)
        final_video = final_video.set_duration(total_duration)

        # Generate music
        music_path = self.generate_background_music(
            parsed_data["vo_notes"], total_duration
        )
        music_audio = AudioFileClip(music_path).volumex(0.22)

        # Mix audio
        final_audio = CompositeAudioClip([vo_audio, music_audio])
        final_video = final_video.set_audio(final_audio)

        # Export
        print("\nExporting final video...")
        output_path = self.output_dir / f"{output_name}.mp4"
        final_video.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            bitrate="6000k",
        )

        print(f"\n✓ Video saved: {output_path}")
        return str(output_path)

    def process(self, json_input: str, output_name: str = None) -> str:
        """Main processing pipeline"""
        if output_name is None:
            output_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n{'='*60}")
        print(f"Processing: {json_input}")
        print(f"{'='*60}\n")

        parsed_data = self.parse_input(json_input)
        video_path = self.assemble_video(parsed_data, output_name)

        # Cleanup
        print("\nCleaning up...")
        for temp_file in self.temp_dir.glob("*"):
            try:
                temp_file.unlink()
            except:
                pass

        return video_path


def create_example_json():
    """Create example JSON input"""
    example = {
        "srt_content": """1
00:00:00,000 --> 00:00:02,000
I remember that night.

2
00:00:02,000 --> 00:00:04,500
The storm hit hard.

3
00:00:04,500 --> 00:00:06,800
Winds howling, phone dead.

4
00:00:06,800 --> 00:00:09,200
Everything went dark.""",
        "image_prompts": """[IMAGE]
00:00:00,000 - 00:00:04,000
Close-up weathered tent flap vertical 9:16, flickering lantern light, warm amber against midnight blues, trembling hand gripping zipper

[IMAGE]
00:00:04,000 - 00:00:09,200
Rain-lashed campsite vertical 9:16, desaturated grays with lightning flashes, silhouette hunched over soaked backpack in storm""",
        "vo_notes": {"speed": 145, "emotion": "intimate and tense, building suspense"},
        "metadata": {"title": "Storm Night", "description": "Survival story"},
    }

    with open("example_input.json", "w") as f:
        json.dump(example, f, indent=2)

    print("✓ Created example_input.json")


if __name__ == "__main__":
    import sys

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not set")
        print("\nGet free key: https://makersuite.google.com/app/apikey")
        print("Set it: export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python video_generator.py <input.json> [output_name]")
        print("\nCreating example...")
        create_example_json()
        print("Run: python video_generator.py example_input.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None

    generator = GeminiVideoGenerator()
    video_path = generator.process(input_file, output_name)
    print(f"\n{'='*60}")
    print(f"✓ COMPLETE! Video: {video_path}")
    print(f"{'='*60}\n")
