# VERITAS — AI-Powered Fake News Detection

**VERITAS** is an AI-powered fact-checking application designed to help users analyze news articles and claims to identify potentially misleading or false information. It uses Google's Gemini API to analyze submitted content and generate AI-assisted assessments.

## Features

- **AI-Powered Analysis:** Analyze news claims and article text using Google's Gemini API.
- **Fact-Checking Assistance:** Generate assessments to help users evaluate the credibility of news content.
- **Simple User Interface:** Interact with the application through a Streamlit-based interface.
- **Report Generation:** Present analysis results in a readable format.
- **Cloud Deployment:** Deploy the application using Streamlit Community Cloud.

> Note: Features such as image, audio, video analysis, evidence retrieval, or PDF report downloads are available only if implemented in the current version of the application.

## Tech Stack

- **Programming Language:** Python
- **Frontend and Application Interface:** Streamlit
- **AI Integration:** Google Gemini API
- **Environment Variables:** python-dotenv
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## Project Structure

```text
VERITAS/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── .streamlit/
    └── config.toml
```

The `.streamlit/config.toml` file is optional and can be used for application configuration. Keep secret credentials out of the repository.

## Getting Started

### Prerequisites

Before running the project, make sure you have:

- Python installed
- Git installed
- A Google Gemini API key
- An internet connection

### 1. Clone the Repository

```bash
git clone https://github.com/eunemi/VERITAS.git
cd VERITAS
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your actual API key.

Get an API key from [Google AI Studio](https://aistudio.google.com/).

**Security:** Never commit your `.env` file or publish your API key on GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this address in your browser to use the application.

## Deploy on Streamlit Community Cloud

1. Push the project code to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with your GitHub account.
4. Create a new app and select the repository and branch.
5. Set the main application file to `app.py`.
6. Add `GEMINI_API_KEY` in the application's Secrets settings.
7. Deploy the application and test the generated URL.

Make sure `requirements.txt` contains all the external Python packages imported by the application.

## How It Works

1. The user submits a news claim or article.
2. The application sends the submitted content to the configured Gemini model.
3. The model analyzes the content according to the application's prompt and available context.
4. The application displays the generated assessment to the user.
5. The user can review the result and use it as a starting point for further verification.

The quality of the assessment depends on the model, input content, prompt design, and availability of reliable supporting evidence.

## Limitations

- AI-generated assessments may contain errors.
- A model's response alone does not prove that a claim is true or false.
- Reliable fact-checking may require independent sources and additional evidence.
- Results can be affected by incomplete, outdated, or misleading input.
- API usage may be subject to rate limits and usage costs.

**Important:** VERITAS should be used as a fact-checking assistance tool, not as a substitute for independent verification from trustworthy sources.

## Future Enhancements

- Integration with trusted fact-checking sources.
- Evidence retrieval and source citations.
- Improved confidence and uncertainty reporting.
- Support for additional media formats.
- Downloadable PDF analysis reports.
- Enhanced error handling and input validation.

These are potential improvements and may not be available in the current version.

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request describing your contribution.

## License

Check the repository's existing license before redistributing or modifying the project. If you are maintaining the original repository, preserve its license and attribution requirements.

---

**VERITAS — Making news verification more accessible through AI.**
