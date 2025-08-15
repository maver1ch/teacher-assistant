# INSTRUTION
Không được code những gì mới và không nằm trong kế hoạc cuả tôi. Khi có đề xuất code thêm những hàm, phần mới, cần phải hỏi ý kiến tôi trước. Hạn chế Error handling và Comment không cần thiết.

CODE REQUIREMENTS:
Always communicate in Vietnamese

## Constants Over Magic Numbers
- No hard-coded values with named constants
- Keep constants at the top of the file

## Smart Comments
- Don't not comment to much. Comment in English
- Use comments to explain why something is done a certain way.

## DRY (Don't Repeat Yourself)
- Extract repeated code into reusable functions
- Share common logic through proper abstraction
- Maintain single sources of truth

## Encapsulation
- Expose clear interfaces
- Move nested conditionals into well-named functions

## KEY NOTES:
- Strictly adhere to the explicitly given instructions. Do not do anything extra.
- Never make medium to large changes based on your own ideas and initiative. Always ask and suggest to me first. => ONLY make changes IF I accepted.
- Work incrementally. Do not try to complete the entire task in one go.
- Do not add comments in code to make notes to me about the changes you made. That goes in the chat not in the code. Only make comments in code as though you are a developer making changes and leaving notes for non-obvious or temporary changes.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

This is a Vietnamese teacher assistant application built with Streamlit for automated grading of student exam submissions. The app processes exam papers and student submissions through OCR, then uses AI (Google Gemini) to analyze questions and grade responses.

### Core Architecture

The application follows a multi-step workflow:

1. **OCR & Exam Setup**: Upload and OCR exam images, create exam in database
2. **Question Analysis**: Use Gemini AI to parse exam into individual questions with difficulty ratings
3. **Submission Processing**: Upload student submissions, OCR text, segment into question-specific answers  
4. **AI Grading**: Grade each submission item using LLM analysis
5. **Report Generation**: Generate comprehensive grading reports

### Key Components

- `app.py`: Main Streamlit application with 5-step workflow
- `database/`: SQLAlchemy models (Exam, Question, Submission, SubmissionItem, Grading) and database manager
- `services/llm_service.py`: Google Gemini integration for question analysis and submission segmentation
- `services/grading_service.py`: Core grading logic using AI
- `services/ocr_service.py`: OCR processing for images

The database uses SQLite with foreign key relationships between exams, questions, submissions, and gradings. Session state management allows jumping between workflow steps and preserves user data.

## Development Commands

### Installation & Setup
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

### Environment Configuration
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Database Schema

The application uses SQLAlchemy with the following key models:
- `Exam`: Contains exam metadata and questions
- `Question`: Individual exam questions with difficulty ratings and knowledge topics  
- `Submission`: Student submissions linked to exams
- `SubmissionItem`: Segmented answers for specific questions
- `Grading`: AI-generated feedback and scores

Data is stored in `data/database.db` (SQLite).

## AI Integration

- **Model**: Google Gemini 2.5 Pro via `google-genai` library
- **Question Analysis**: Parses Vietnamese exam text, extracts individual questions with difficulty (1-10 scale)
- **Answer Segmentation**: Matches student responses to specific questions using semantic analysis
- **Grading**: Provides detailed feedback in Vietnamese with knowledge gap identification

## Session State Management

The Streamlit app uses extensive session state to:
- Maintain workflow position (`current_step`)
- Store OCR results and parsed questions
- Track exam/submission IDs for database operations
- Allow navigation between steps via sidebar

## Key Features

- Multi-image OCR support for both exams and submissions
- Real-time LaTeX preview with $/$$$ syntax
- Vietnamese language support throughout
- Database persistence with ability to resume work
- AI-powered question difficulty assessment
- Comprehensive grading reports with knowledge gap analysis