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

This is a Vietnamese teacher assistant application built with Streamlit for automated grading of student exam submissions. The app processes exam papers and student submissions through OCR, then uses AI (OpenAI models) to analyze questions and grade responses.

### Core Architecture

The application follows a 6-step workflow:

1. **OCR & Exam Setup**: Upload and OCR exam images, create exam in database
2. **Question Analysis**: Use OpenAI models to parse exam into individual questions with difficulty ratings
3. **Solution Generation**: Create standard solutions and grading rubrics for each question
4. **Submission Processing**: Upload student submissions, OCR text, segment into question-specific answers  
5. **AI Grading**: Grade each submission item by comparing with standard solutions
6. **Report Generation**: Generate comprehensive grading reports and export functionality

### Key Components

- `app.py`: Main Streamlit application with 6-step workflow and sidebar navigation
- `database/`: SQLAlchemy models and database manager
  - `models.py`: Database models (Exam, Question, Submission, SubmissionItem, Grading, QuestionSolution, SubmissionReport)
  - `db_manager.py`: Database operations and migrations
- `services/`: Core business logic services
  - `llm_service.py`: OpenAI integration for question analysis and submission segmentation
  - `grading_service.py`: Core grading logic comparing student answers with solutions
  - `ocr_service.py`: OCR processing for images using OpenAI vision models
  - `solution_service.py`: Generate standard solutions and grading rubrics

The database uses SQLite with foreign key relationships. Session state management allows jumping between workflow steps and preserves user data.

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
OPENAI_API_KEY=your_openai_api_key_here
```

## Database Schema

The application uses SQLAlchemy with the following key models:
- `Exam`: Contains exam metadata and original text
- `Question`: Individual exam questions with difficulty ratings, knowledge topics, order_index, and part_label
- `QuestionSolution`: Standard solutions with reasoning approach and final answers
- `Submission`: Student submissions linked to exams with student names
- `SubmissionItem`: Segmented answers for specific questions with position tracking
- `Grading`: AI-generated feedback comparing student answers with solutions (knowledge_gaps, calculation_logic_errors, llm_feedback, is_correct)
- `SubmissionReport`: Generated reports for students

Data is stored in `data/database.db` (SQLite).

## AI Integration

- **Models**: OpenAI models including o4-mini, gpt-4.1-mini for different tasks
- **Question Analysis**: Parses Vietnamese exam text, extracts individual questions with difficulty (1-10 scale) and knowledge topics
- **Solution Generation**: Creates standard solutions, final answers, and grading rubrics
- **Answer Segmentation**: Matches student responses to specific questions using skeleton-based approach
- **Grading**: Compares student answers with standard solutions, provides detailed feedback in Vietnamese with knowledge gap identification

## Session State Management

The Streamlit app uses extensive session state to:
- Maintain workflow position (`current_step`)
- Store OCR results and parsed questions
- Track exam/submission IDs for database operations
- Allow navigation between steps via sidebar with data selection from DB
- Preserve editor content and segmented items

## Key Features

- 6-step workflow with ability to jump between steps
- Multi-image OCR support for both exams and submissions
- Real-time LaTeX preview with $/$$$ syntax support
- Vietnamese language support throughout
- Database persistence with ability to resume work from any step
- AI-powered question difficulty assessment and knowledge topic extraction
- Standard solution generation with grading rubrics
- Comprehensive grading with knowledge gap analysis and calculation error detection
- Export functionality for gradings and student summaries
- Editable segmentation results with LaTeX preview