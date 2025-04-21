from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1. Функция расчета индекса цифровизации
def calculate_digitalization_index(data):
    scores = {
        'online_sales': min(data['online_sales'] / 100, 1) * 0.15,
        'automated_processes': min(data['automated_processes'] / 100, 1) * 0.12,
        'online_payments': 1 * 0.10 if data['online_payments'] else 0,
        'task_completion_time': (1 - min(data['task_time'] / 240, 1)) * 0.10,
        'support_response': (1 - min(data['support_time'] / 60, 1)) * 0.09,
        'e_documents': min(data['e_docs'] / 100, 1) * 0.08,
        'mobile_app': 1 * 0.07 if data['mobile_app'] else 0,
        'remote_work': min(data['remote_workers'] / 100, 1) * 0.06,
        'digital_skills': min(data['skills_score'] / 10, 1) * 0.05,
        'chatbots': 1 * 0.05 if data['chatbots'] else 0,
        'digital_marketing': 1 * 0.04 if data['digital_marketing'] else 0,
        'online_training': 1 * 0.03 if data['online_training'] else 0,
        'big_data': 1 * 0.03 if data['big_data'] else 0,
        'email_automation': 1 * 0.02 if data['email_automation'] else 0,
        'e_passes': 1 * 0.01 if data['e_passes'] else 0
    }
    total_score = sum(scores.values())
    return {
        'digitalization_index': round(total_score * 100, 1),
        'components': scores
    }

# 2. Настройка модели
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# 3. Данные компании Да=True, Нет=False
company_data = {
    'online_sales': 45,
    'automated_processes': 60,
    'online_payments': False,
    'task_time': 20,
    'support_time': 11,
    'e_docs': 80,
    'mobile_app': True,
    'remote_workers': 30,
    'skills_score': 7.5,
    'chatbots': True,
    'digital_marketing': True,
    'online_training': True,
    'big_data': False,
    'email_automation': True,
    'e_passes': True
}

# 4. Расчет индекса
result = calculate_digitalization_index(company_data)

# 5. Формирование промпта
analysis_prompt = f"""
Проанализируй уровень цифровизации предприятия по следующим критериям. 
Для каждого критерия предоставь способы улучшения в строгом формате:

[Номер]. [Название критерия]
Способы улучшить: [Подробно описаные решения] 


Критерии и текущие значения:

1. Доля онлайн-продаж: {company_data['online_sales']}%
2. Уровень автоматизации процессов: {company_data['automated_processes']}%
3. Наличие онлайн-платежей: {'Да' if company_data['online_payments'] else 'Нет'}
4. Среднее время выполнения задач: {company_data['task_time']} минут
5. Время ответа поддержки: {company_data['support_time']} минут
6. Доля электронных документов: {company_data['e_docs']}%
7. Наличие мобильного приложения: {'Есть' if company_data['mobile_app'] else 'Нет'}
8. Доля удаленных сотрудников: {company_data['remote_workers']}%
9. Уровень цифровых навыков: {company_data['skills_score']}/10
10. Использование чат-ботов: {'Есть' if company_data['chatbots'] else 'Нет'}
11. Применение цифрового маркетинга: {'Используется' if company_data['digital_marketing'] else 'Нет'}
12. Доступность онлайн-обучения: {'Доступно' if company_data['online_training'] else 'Нет'}
13. Использование Big Data: {'Используется' if company_data['big_data'] else 'Нет'}
14. Автоматизация email-рассылок: {'Автоматизированы' if company_data['email_automation'] else 'Нет'}
15. Система электронных пропусков: {'Внедрена' if company_data['e_passes'] else 'Нет'}

Общий индекс цифровизации: {result['digitalization_index']}/100

"""

# 6. Генерация и вывод рекомендаций
print(f"Индекс цифровизации предприятия: {result['digitalization_index']}/100")
print("\nАнализ и рекомендации по критериям:\n")

inputs = tokenizer(analysis_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# Постобработка и вывод результатов
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
recommendations = full_response.split("Критерии и текущие значения:")[-1].strip()
print(recommendations)

# 7. Очистка памяти
del model
del tokenizer
torch.cuda.empty_cache()