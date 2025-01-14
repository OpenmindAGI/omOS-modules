# omOS Modules

A Python package containing ML modules for omOS, including speech processing, utilities, and vision-language capabilities.

## Installation

This project uses Poetry for dependency management.

### Step 1: Install Poetry

If you haven’t already, install Poetry by following the official installation guide.

### Step 2: Install the Package

Run the following command to install the dependencies and set up the environment:

```bash
poetry install
```

## Modules

### omOS Speech

Speech processing module providing the following features:

- Audio input/output handling (`audio_input_stream.py`)
- NVIDIA Riva ASR/TTS integration
- WebSocket streaming support

### omOS Utils

Common utility functions for:

* HTTP client/server
* WebSocket handling
* Message formatting

### omOS VLM

Vision-Language Module offering:

* Video stream processing
* Language model integration
* NVIDIA Nano LLM support

## Development

### Set Up the Development Environment

1. Clone the repository:

	 ```bash
	 git clone https://github.com/openmind-org/omOS-modules.git
	 cd omOS-modules
	 ```

2. Install dependencies using Poetry:

	 ```bash
	 poetry install
	 ```

### Adding Dependencies

To add a new dependency, use:

```bash
poetry add <package_name>
```

## Code Structure

* *omOS_speech*: Contains speech processing modules.
* *omOS_utils*: Utility functions for HTTP, WebSocket, and message handling.
* *omOS_vlm*: Vision-language module with video stream and LLM support.

## Contributing

We welcome contributions! To contribute, follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](MIT) file for details.

## Authors

Developed and maintained by [**openmind.org**](openmind.org).