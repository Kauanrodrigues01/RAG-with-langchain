# 🧠 RAG - Retrieval-Augmented Generation com Langchain

Este repositório contém experimentos e implementações que desenvolvi enquanto aprendia sobre **RAG (Retrieval-Augmented Generation)**, uma técnica que combina modelos de linguagem (LLMs) com bases de conhecimento externas (como arquivos PDF, CSV ou bancos vetoriais) para fornecer respostas mais precisas, confiáveis e contextualizadas.

## 📂 Estrutura do Repositório

* **📁 01-rag-com-pdf/**
  Exemplo de aplicação RAG utilizando arquivos **PDF** como fonte de dados.
  Utiliza loaders, text splitters e armazena os embeddings para consultas futuras.

* **📁 02-persistindo-vector-store/**
  Demonstra como **persistir uma vector store** (base vetorial) para reutilização, economizando processamento e melhorando o tempo de resposta.

* **📁 03-usando-vector-store-persistida/**
  Mostra como **carregar uma vector store já persistida** para realizar consultas com Langchain + LLM, sem reprocessar os dados.

* **📁 04-rag-com-csv/**
  Projeto de RAG utilizando **dados estruturados em CSV** como fonte de conhecimento. Ideal para bases tabulares e análises específicas.

## 🛠️ Tecnologias Utilizadas

* [Langchain](https://www.langchain.com/)
* [OpenAI API](https://platform.openai.com/)
* [ChromaDB](https://www.trychroma.com/) para armazenamento vetorial
* [Python](https://www.python.org/)

## ⚙️ Como Rodar os Exemplos

1. Clone o repositório:

   ```bash
   git clone https://github.com/Kauanrodrigues01/RAG-with-langchain.git
   cd RAG-with-langchain
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS/GitBash
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Adicione suas variáveis de ambiente em um arquivo `.env`:

   ```
   OPENAI_API_KEY=your_openai_key_here
   ```

5. Execute os exemplos desejados:

   ```bash
   python 01-rag-com-pdf/main.py
   ```

## 💡 O Que É RAG?

**Retrieval-Augmented Generation (RAG)** é uma técnica que permite que modelos de linguagem consultem fontes externas (como documentos, bancos de dados ou APIs) durante a geração de texto. Isso permite:

* Redução de alucinação por parte dos modelos
* Respostas mais factuais e baseadas em dados reais
* Capacidade de responder perguntas específicas sobre contextos personalizados

## 📌 Objetivo do Repositório

O foco deste repositório é o **aprendizado prático**. Não se trata de um projeto de produção, mas sim de uma **coleção de experimentos e boas práticas** com RAG, Langchain e armazenamento vetorial.
