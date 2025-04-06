import os
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from openai_model import OpenAIModel
from metric import Recall
from openai import OpenAI
import tiktoken
import os
from query_expansion import LLMQueryExpander
from query_expansion import DictionaryQueryExpander
from reranker import LLMReranker
from reranker import CrossEncoderReranker
from meta_data_provider import MetaDataProvider



client = OpenAI()
link_repo = "https://github.com/viarotel-org/escrcpy"
limit = 8191
rate_limit = 29000
length_threshold = 5
enc = tiktoken.get_encoding("cl100k_base")
models = ["text-embedding-3-small", "text-embedding-ada-002", "text-embedding-3-large"]

# Initialize objects
MetaDataProvider_instance = MetaDataProvider()
LLMQueryExpander_instance = LLMQueryExpander(model="gpt-4o")
DictionaryQueryExpander_instance = DictionaryQueryExpander(1)
LLMReranker_instance = LLMReranker(model_name="gpt-4o", limit=limit, rate_limit=rate_limit, shortening_token=500)
CrossEncoderReranker_instance = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

# Initialize model objects
object1 = OpenAIModel(link_repo, models[0], encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
object1.repo_path = "escrcpy"
object2 = OpenAIModel(link_repo, models[1], encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
object2.repo_path = "escrcpy"
object3 = OpenAIModel(link_repo, models[2], encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
object3.repo_path = "escrcpy"

global_object = None

console = Console()


cancel_flag = False


def print_header():
    console.print(
        Panel(
            "[bold magenta]Welcome to the Example RAG System![/bold magenta] ([italic blue]Type HELP for a list of commands[/italic blue])",
            border_style="magenta",
        )
    )





def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")
    print_header()





def get_input(prompt_text):
    user_input = Prompt.ask(prompt_text)
    return f"{user_input}"





def start_ui():
    global global_object
    global cancel_flag
    clear_screen()
    while True:
        user_input = Prompt.ask("\n[bold white]Enter command[/bold white]")




        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold magenta]Hope to see you soon![/bold magenta]")
            break




        elif user_input.lower() == "price":
            while True:
                    repo_url = get_input("[bold white]Enter GitHub repo URL[/bold white]")
                    if repo_url.lower() == "cancel":
                        cancel_flag = True
                        break
                    model = get_input("[bold white]Enter model number\n 1. text-embedding-3-small \n 2. text-embedding-ada-002 \n 3. text-embedding-3-large[/bold white]\n insert number here:")
                    if model.lower() == "cancel":
                        cancel_flag = True
                        break
                    match model:
                        case "1":
                            model = models[0]
                        case "2":
                            model = models[1]
                        case "3":
                            model = models[2]
                        case _:
                            console.print("[bold magenta]Invalid model number![/bold magenta]")
                            continue
                    model_temp = OpenAIModel(repo_url, model, encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
                    model_temp.clone_repo()
                    break
            if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
            while True:
                chunk_size = get_input("[bold white]Enter chunk size (0 < size <= 8191)[/bold white]")
                if chunk_size.lower() == "cancel":
                        cancel_flag = True
                        break
                if chunk_size is None:
                    return
                if chunk_size.isdigit() and int(chunk_size) > 0 and int(chunk_size) <= 8191:
                    chunk_size = int(chunk_size)
                    break 
                console.print("[bold magenta]Invalid chunk size![/bold magenta]")
            if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
            while True:
                overlap_size = get_input("[bold white]Enter overlapp size[/bold white]")
                if overlap_size.lower() == "cancel":
                        cancel_flag = True
                        break
                if overlap_size is None:
                    return
                if overlap_size.isdigit() and int(overlap_size) > 0:
                    overlap_size= int(overlap_size)
                    break
            if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
            while True:
                    metadata = get_input("[bold white]Do you want to use metadata? (yes/no)[/bold white]")
                    if metadata.lower() == "cancel":
                        cancel_flag = True
                        break
                    if metadata.lower() == "yes":
                        metadata = MetaDataProvider_instance
                        break
                    elif metadata.lower() == "no":
                        metadata = None
                        break 
            if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
            cost = model_temp.build_cost(mutiple_chunks_per_file=True, chunk_size=chunk_size, overlap_size=overlap_size, metadata=metadata)
            console.print(f"[bold yellow]Approximate cost of building the index: {cost} USD[/bold yellow]")




        elif user_input.lower() == "index":
            try:
                while True:
                    repo_url = get_input("[bold white]Enter GitHub repo URL[/bold white]")
                    if repo_url.lower() == "cancel":
                        cancel_flag = True
                        break
                    model = get_input("[bold white]Enter model number\n 1. text-embedding-3-small \n 2. text-embedding-ada-002 \n 3. text-embedding-3-large[/bold white]\n insert number here")
                    if model.lower() == "cancel":
                        cancel_flag = True
                        break
                    match model:
                        case "1":
                            model = models[0]
                        case "2":
                            model = models[1]
                        case "3":
                            model = models[2]
                        case _:
                            console.print("[bold magenta]Invalid model number![/bold magenta]")
                            continue
                    global_object = OpenAIModel(repo_url, model, encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
                    global_object.clone_repo()
                    break
                if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue

                while True:
                    chunk_size = get_input("[bold white]Enter chunk size (0 < size <= 8191)[/bold white]")
                    if chunk_size.lower() == "cancel":
                        cancel_flag = True
                        break
                    if chunk_size.isdigit() and int(chunk_size) > 0 and int(chunk_size) <= 8191:
                        chunk_size = int(chunk_size)
                        break 
                    console.print("[bold magenta]Invalid chunk size![/bold magenta]")
                if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
                while True:
                    overlap_size = get_input("[bold white]Enter overlapp size[/bold white]")
                    if overlap_size.lower() == "cancel":
                        cancel_flag = True
                        break
                    if overlap_size.isdigit() and int(overlap_size) > 0:
                        overlap_size= int(overlap_size)
                        break 
                if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
                while True:
                    metadata = get_input("[bold white]Do you want to use metadata? (yes/no)[/bold white]")
                    if metadata.lower() == "cancel":
                        cancel_flag = True
                        break
                    if metadata.lower() == "yes":
                        metadata = MetaDataProvider_instance
                        break
                    elif metadata.lower() == "no":
                        metadata = None
                        break
                    console.print("[bold magenta]Invalid input![/bold magenta]")
                if cancel_flag:
                    cancel_flag = False
                    console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                    continue
                global_object.create_index(
                    index_name="faiss_index_openai_large_meta_data",
                    file_store_name="file_paths_openai_large_meta_data",
                    mutiple_chunks_per_file=True,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    metadata=metadata
                )
                console.print("[bold yellow]Creating...[/bold yellow]")
                console.print("[bold yellow]Indexing complete![/bold yellow]")

            except KeyboardInterrupt:
                console.print("\n[bold magenta]Indexing interrupted by user.[/bold magenta]")








        elif user_input.lower() == "query":
            if global_object is None:
                console.print("[bold magenta]Please index a repository first![/bold magenta]")
                continue

            query = get_input("[bold white]Enter your query[/bold white]")
            if query is None:
                return
            while True:
                reranker = get_input("[bold white]Do you want to use a reranker? (yes/no)[/bold white]")
                if reranker.lower() == "cancel":
                    cancel_flag = True
                    break
                if reranker.lower() == "yes":
                    reranker = get_input ("[bold white]Enter reranker type (number)\n 1. LLM \n 2. CrossEncoder[/bold white] \n insert number here")
                    if reranker.lower() == "1":
                        reranker = LLMReranker_instance
                    elif reranker.lower() == "2":
                        reranker = CrossEncoderReranker_instance
                    else:
                        reranker = None
                        console.print("[bold magenta]Invalid input None was chosen![/bold magenta]")
                        break
                    while True:
                        number_to_rerank = get_input("[bold white]Enter number to rerank (number how many files rereanker will take into account (reccomended 15))[/bold white]")
                        if number_to_rerank.lower() == "cancel":
                            cancel_flag = True
                            break
                        if number_to_rerank.isdigit() and int(number_to_rerank) > 0:
                            number_to_rerank = int(number_to_rerank)
                            break 
                        console.print("[bold magenta]Invalid input![/bold magenta]")
                    break
                elif reranker.lower() == "no":
                    reranker = None
                    number_to_rerank = None
                    break
                else:
                    console.print("[bold magenta]Invalid input![/bold magenta]")
                    continue
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue
            while True:
                query_expansion = get_input("[bold white]Do you want to use query expansion? (yes/no)[/bold white]")
                if query_expansion.lower() == "cancel":
                    cancel_flag = True
                    break
                if query_expansion.lower() == "yes":
                    query_expansion = get_input ("[bold white]Enter query expansion type (number)\n 1. LLM \n 2. Dictionary \n insert number here[/bold white]")
                    if query_expansion.lower() == "1":
                        query_expansion = LLMQueryExpander_instance
                        break
                    elif query_expansion.lower() == "2":
                        query_expansion = DictionaryQueryExpander_instance
                        break
                    else:
                        query_expansion = None
                        console.print("[bold magenta]Invalid input None was chosen![/bold magenta]")
                elif query_expansion.lower() == "no":
                    query_expansion = None
                    break
                else:
                    console.print("[bold magenta]Invalid input![/bold magenta]")
                    continue
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue
            while True:
                top_k = get_input("[bold white]Enter number of top k results[/bold white]")
                if top_k.lower() == "cancel":
                    cancel_flag = True
                    break
                if top_k.isdigit() and int(top_k) > 0:
                    top_k = int(top_k)
                    break 
                console.print("[bold magenta]Invalid input![/bold magenta]")
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue
            while True:
                summary = get_input("[bold white]Do you want to display summary? (yes/no)[/bold white]")
                if summary.lower() == "cancel":
                    cancel_flag = True
                    break
                if summary.lower() == "yes":
                    summary = True
                    break
                elif summary.lower() == "no":
                    summary = False
                    break
                console.print("[bold magenta]Invalid input![/bold magenta]")
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue
            while True:
                show_tokens = get_input("[bold white]Do you want to display token usage? (yes/no)[/bold white]")
                if show_tokens.lower() == "cancel":
                    cancel_flag = True
                    break
                if show_tokens.lower() == "yes":
                    show_tokens = True
                    break
                elif show_tokens.lower() == "no":
                    show_tokens = False
                    break
                console.print("[bold magenta]Invalid input![/bold magenta]")
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue
            while True:
                show_time = get_input("[bold white]Do you want to diplay time? (yes/no)[/bold white]")
                if show_time.lower() == "cancel":
                    cancel_flag = True
                    break
                if show_time.lower() == "yes":
                    show_time = True
                    break
                elif show_time.lower() == "no":
                    show_time = False
                    break
                console.print("[bold magenta]Invalid input![/bold magenta]")
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue

                
            console.print("[bold yellow]Searching...[/bold yellow]")
            results = global_object.search([query], top_k=top_k, rerank=reranker, number_to_rerank=number_to_rerank, query_expansion=query_expansion, summary=summary, token_usage=show_tokens, time=show_time)
            console.print("[bold yellow]Searching finished...[/bold yellow]")
            print(f"")
           

            console.print(f"[bold blue]top {top_k} results[/bold blue]")
            for file in results[0][0]:
                console.print(f"{file}")
            if summary:
                console.print(f"\n[bold blue]Summary:[/bold blue]\n{results[1][0]}\n")
            if show_tokens:
                console.print(f"[bold blue]Token Usage: {results[2][0]}[/bold blue]\n")
            if show_time:
                console.print(f"[bold blue]Time: {results[3]}[/bold blue]\n")
        elif user_input.lower() == "evaluate":
            while True:
                if global_object is None:
                    console.print("[bold magenta]Please index a repository first![/bold magenta]")
                    break
                print("evaluted wihout reranker and query expansion due to high latency")
                recall_number = get_input("[bold white]Enter recall number:")
                if recall_number.lower() == "cancel":
                    cancel_flag = True
                    break
                if recall_number.isdigit() and int(recall_number) > 0:
                    recall_number = int(recall_number)
                    recall = Recall(recall_number,  data_path='escrcpy-commits-generated.json')
                    print(f"recall number: {recall_number}")
                    print(global_object.evaluate(top_k=recall_number, metric=recall, query_expansion=None, rerank=None, token_usage=True, time=True)[1])
                    break
            if cancel_flag:
                cancel_flag = False
                console.print("[bold magenta]Operation cancelled.[/bold magenta]")
                continue









        elif user_input.lower() == "clear":
            clear_screen()







        elif user_input.lower() == "help":
            console.print("\n[bold magenta]Available Commands:[/bold magenta]")
            console.print("  [bold blue]cancel[/bold blue] - cancel any operation and go back to the main menu")
            console.print("  [bold blue]price[/bold blue] - approximate price of the building index")
            console.print("  [bold blue]index[/bold blue] - Index a GitHub repo")
            console.print("  [bold blue]query[/bold blue] - Start querying the data")
            console.print("  [bold blue]evaluate[/bold blue]  - show evaluation of the model")
            console.print("  [bold blue]exit[/bold blue]  - Quit the system")
            console.print("  [bold blue]clear[/bold blue] - Clear the screen")
            console.print("  [bold blue]help[/bold blue]  - Show available commands")
            

        else:
            console.print("[bold magenta]Unknown command. Type HELP for options.[/bold magenta]")

if __name__ == "__main__":
    start_ui()
