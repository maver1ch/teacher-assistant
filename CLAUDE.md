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

This is a Vietnamese teacher assistant application built with Streamlit for automated grading of student exam submissions. The app processes exam papers and student submissions through AI analysis, then provides intelligent grading and feedback.

### Core Architecture

The application follows a 4-step workflow (displayed as steps 1, 2, 3, 4 in the UI):

1. **Exam Analysis (Step 1)**: Upload and analyze exam images using OpenAI vision models, parse into individual questions with difficulty ratings and knowledge topics
2. **Solution Generation (Step 2)**: Create standard solutions with reasoning approaches and grading rubrics for each question  
3. **Submission Processing (Step 3)**: Upload student submission images, process and segment into question-specific answers using skeleton-based matching
4. **AI Grading (Step 4)**: Grade each submission item by comparing with standard solutions, identify knowledge gaps and calculation errors, generate reports

### Key Components

- `app.py`: Main Streamlit application with 4-step workflow and sidebar navigation
- `database/`: SQLAlchemy models and database manager
  - `models.py`: Database models (Exam, Question, Submission, SubmissionItem, Grading, QuestionSolution, SubmissionReport)
  - `db_manager.py`: Database operations and migrations
- `services/`: Core business logic services
  - `exam_analyzer.py`: OpenAI vision models for OCR and question analysis from exam images
  - `grading_service.py`: Core grading logic comparing student answers with solutions using GPT-5-mini for advanced reasoning
  - `submission_processor.py`: Process and segment student submissions using skeleton-based approach
  - `solution_service.py`: Generate standard solutions and grading rubrics using GPT-5-mini
  - `performance_analyzer.py`: Advanced analysis to group common mistakes and knowledge gaps
- `utils/`: Utility modules and configurations
  - `config.py`: Application configuration including model settings and API keys
  - `schemas.py`: JSON schemas for OpenAI API responses
  - `data_models.py`: Data models for internal processing (GradingResult, SolutionResult, QuestionLite)
  - `prompts.py`: Comprehensive system prompts for various AI tasks
  - `llm_logger.py`: LLM API call logging with cost tracking
- `export_gradings.py`: Standalone script for exporting grading results to CSV
- `delete_gradings.py`: CLI tool for clearing grading data

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
- `Exam`: Contains exam metadata and name (no original text stored)
- `Question`: Individual exam questions with difficulty ratings, knowledge topics (JSON), order_index, and part_label
- `QuestionSolution`: Standard solutions with reasoning approach and final answers
- `Submission`: Student submissions linked to exams with student names and optional original_text
- `SubmissionItem`: Segmented answers for specific questions with position tracking
- `Grading`: AI-generated feedback with knowledge gaps analysis, calculation/logic errors, correctness assessment, and final scores
- `SubmissionReport`: Generated markdown reports for students

Data is stored in `data/database.db` (SQLite). The database includes automatic migrations and foreign key relationships.

## AI Integration

- **Models**: Multiple OpenAI models configured for different tasks:
  - `gpt-4.1-mini`: Exam analysis, submission processing, grouping analysis, and basic grading
  - `gpt-5-mini`: Solution generation and advanced grading with reasoning
  - Temperature set to 0.1 for consistent results
- **API Configuration**: OpenAI client initialized per service with environment variable OPENAI_API_KEY
- **Vision OCR**: Uses OpenAI vision models to extract text from exam and submission images with base64 encoding
- **Question Analysis**: Parses Vietnamese exam content, extracts individual questions with difficulty (1-10 scale), knowledge topics (3-5 tags required), and hierarchical labels
- **Solution Generation**: Creates comprehensive solutions with reasoning approaches, final answers, and difficulty assessment using context from related questions
- **Skeleton-based Segmentation**: Matches student responses to specific questions using pre-defined question structure
- **Advanced Grading**: Analyzes student answers against standard solutions using GPT-5-mini, identifies knowledge gaps, calculation errors, and provides Vietnamese feedback with detailed error categorization
- **Performance Analysis**: Groups common mistakes and knowledge gaps across submissions

## Session State Management

The Streamlit app uses extensive session state to:
- Maintain workflow position (`current_step`)
- Store OCR results and parsed questions
- Track exam/submission IDs for database operations
- Allow navigation between steps via sidebar with data selection from DB
- Preserve editor content and segmented items

## Key Features

- 4-step workflow with ability to jump between steps via sidebar navigation
- Multi-image OCR support for both exams and submissions using OpenAI vision models
- Real-time LaTeX preview with $/$$$ syntax support and interactive line clicking
- Vietnamese language support throughout all UI and AI responses
- Database persistence with ability to resume work from any step
- AI-powered question difficulty assessment (1-10 scale) and knowledge topic extraction (3-5 tags)
- Standard solution generation with grading rubrics using context from related questions
- Comprehensive grading with knowledge gap analysis and calculation error detection
- Performance analysis to group common mistakes across submissions
- Export functionality for detailed gradings and student summaries to CSV
- Editable segmentation results with LaTeX preview and validation
- Report generation with markdown export
- LLM API call logging with cost tracking
- Database migration support and CLI tools for data management

## Technical Implementation

### Data Flow
1. **Image Processing**: Multi-image uploads → Base64 encoding → OpenAI Vision API
2. **Question Parsing**: Vision OCR → JSON schema validation → Database storage
3. **Solution Generation**: Question context → GPT-5-mini reasoning → Solution storage
4. **Submission Segmentation**: Skeleton creation → Vision matching → Editable results
5. **Grading**: Solution comparison → GPT-5-mini analysis → Detailed feedback
6. **Reporting**: Data aggregation → Report generation → Export functionality

### Error Handling
- JSON schema validation for all AI responses
- Graceful fallbacks for API failures
- Data validation with type conversion
- Database transaction safety

### Performance Considerations
- Efficient database queries with joins and indexing
- Session state management for large datasets
- Streaming responses for long-running operations
- Cost-optimized model selection per task