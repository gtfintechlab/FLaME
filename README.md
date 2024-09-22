`glennmatlin[at]gatech[dot]edu`

## Project Setup

### Creating and Activating the Virtual Environment

To create the virtual environment in the project root and install the required packages, follow these steps:

1. **Create the virtual environment**:
    ```sh
    python -m venv .venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\.venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source .venv/bin/activate
        ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

### API keys
To configure your API keys, follow these steps:

1. **Create a `.env` file**:
    - You can create a new `.env` file in the project root directory **OR** copy the provided `.env.sample` file and rename it to `.env`.

2. **Modify the `.env` file**:
    - Open the `.env` file in a text editor.
    - Add your API keys in the following format:
      ```
      API_KEY_NAME=your_api_key_value
      ```
    - Replace `API_KEY_NAME` with the actual name of the API key and `your_api_key_value` with your actual API key.

3. **Save the `.env` file**:
    - Ensure the file is saved in the project root directory.

Example:
```
TOGETHER_API_KEY=foo
OPENAI_API_KEY=bar
ANTHROPIC_API_KEY=baz
```