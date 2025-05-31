import gradio as gr
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from transformers import pipeline
import torch
import logging
from typing import Dict, List, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Состояние графа
class ResearchState(Dict):
    query: str
    results: List[Dict]
    summary: str
    links: List[Dict]


# Агент для исследований
class ResearchAgent:
    def __init__(self):
        self.summarizer = self._init_model()

    def _init_model(self):
        try:
            device = -1  # Принудительно используем CPU
            model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            logger.info("Модель успешно загружена")
            return model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return None

    def _clean_search_results(self, results: List[Dict]) -> List[Dict]:
        seen_urls = set()
        cleaned = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get('href') or result.get('url') or ''
            title = result.get('title', '')
            content = result.get('body') or result.get('description') or result.get('content', '')
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            # Извлекаем автора/источник из заголовка
            source = title.split('—')[-1].strip() if '—' in title else title.split('-')[-1].strip()

            cleaned.append({
                'title': str(title)[:200],
                'content': str(content)[:500],
                'url': str(url),
                'source': source[:100]  # Ограничиваем длину источника
            })
        return cleaned[:5]  # Ограничиваем 5 результатами

    def search(self, query: str) -> List[Dict]:
        if not query or len(query.strip()) < 3:
            return []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))
                return self._clean_search_results(results)
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def summarize(self, text: str) -> str:
        if not text or not self.summarizer:
            return "Нет данных для суммаризации"
        text = text.strip()
        if len(text.split()) < 10:
            return text
        try:
            word_count = len(text.split())
            max_len = min(130, max(30, word_count // 2))
            min_len = min(30, max_len // 2)
            result = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            return result[0]["summary_text"] if result else "Не удалось создать краткое содержание"
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
            return "Ошибка при создании краткого содержания"


agent = ResearchAgent()


# Узлы графа
def search_node(state: ResearchState) -> ResearchState:
    query = state.get("query", "").strip()
    logger.info(f"Выполняется поиск по запросу: {query}")
    results = agent.search(query)
    return {"results": results}


def summarize_node(state: ResearchState) -> ResearchState:
    results = state.get("results", [])
    if not results:
        return {
            "summary": "Результаты не найдены.",
            "links": []
        }

    # Собираем все тексты для создания общей выжимки
    combined_text = ""
    valid_links = []
    for result in results[:5]:  # Ограничиваем 5 результатами
        if not isinstance(result, dict):
            continue

        text = f"{result.get('title', '')}\n{result.get('content', '')}"
        combined_text += f"\n\n{text}"

        valid_links.append({
            'title': result.get('title', 'Ссылка').split('—')[0].strip(),
            'url': result.get('url', '#'),
            'source': result.get('source', 'Неизвестный источник')
        })

    # Создаем общую краткую выжимку
    summary = agent.summarize(combined_text.strip())

    return {
        'summary': summary if summary else 'Нет результатов',
        'links': valid_links[:5]  # Ограничиваем 5 ссылками
    }


# Построение графа
workflow = StateGraph(ResearchState)

workflow.add_node("search", search_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge("search", "summarize")
workflow.set_entry_point("search")

app = workflow.compile()


# Интерфейс Gradio
def run_agent(query: str) -> tuple:
    try:
        if not query or len(query.strip()) < 3:
            return "Запрос должен содержать минимум 3 символа.", ""

        output = app.invoke({"query": query.strip()})
        summary = output.get('summary', 'Краткое содержание недоступно')
        links = output.get('links', [])[:5]  # Ограничиваем 5 ссылками

        # Форматируем ссылки в HTML
        formatted_links = []
        for link in links:
            if isinstance(link, dict) and link.get('url'):
                title = link.get('title', 'Без названия')
                source = link.get('source', 'Неизвестный источник')
                url = link['url']
                formatted_links.append(
                    f'<div style="margin-bottom: 8px;">'
                    f'<a href="{url}" target="_blank" style="text-decoration: none; color: #1E90FF; font-weight: bold;">'
                    f'{title}</a> — <span style="color: #555;">{source}</span>'
                    f'</div>'
                )

        return (
            summary,
            ''.join(formatted_links) if formatted_links else "Источники не найдены"
        )
    except Exception as e:
        logger.error(f"Ошибка агента: {e}", exc_info=True)
        return f"Ошибка при обработке запроса: {str(e)}", ""


# Тема и логотип
logo_html = """
<div style='text-align:center; margin-bottom: 20px;'>
    <img src='https://cdn-icons-png.flaticon.com/512/1055/1055645.png' width='80'/>
    <h1 style='font-family: "Poppins", sans-serif; color: #1E90FF;'>Агент DeepResearch</h1>
    <p style='color: #555;'>Введите вопрос — получите краткий ответ с источниками</p>
</div>
"""

custom_css = """
body {
    font-family: 'Poppins', sans-serif;
    background-color: #f8f9fa;
}
input[type=text] {
    border-radius: 8px;
    padding: 10px;
    font-size: 16px;
    border: 1px solid #ccc;
}
button {
    background-color: #1E90FF !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
}
.markdown-body {
    font-size: 16px;
    line-height: 1.6;
    padding: 15px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.output-html {
    padding: 15px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
"""

soft_blue_theme = gr.themes.Soft(primary_hue="blue", spacing_size="sm", radius_size="lg")

# Запуск
with gr.Blocks(theme=soft_blue_theme, css=custom_css) as demo:
    gr.HTML(logo_html)
    gr.Interface(
        fn=run_agent,
        inputs=gr.Textbox(label="🔍 Введите ваш вопрос", placeholder="Например: Егор Крид"),
        outputs=[
            gr.Markdown(label="📝 Краткое содержание"),
            gr.HTML(label="🔗 Источники")
        ],
        allow_flagging="never",
        submit_btn="🔎 Поиск",
        clear_btn="🔄 Очистить"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)