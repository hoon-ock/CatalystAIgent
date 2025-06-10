from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_anthropic import ChatAnthropic
from pydantic.v1 import BaseModel, Field, validator
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek 
from typing import List, Optional, Union
import sys
from pathlib import Path
custom_path = Path("fairchem-forked/src").resolve()
if str(custom_path) not in sys.path:
    sys.path.insert(0, str(custom_path))
from fairchem.data.oc.core import Adsorbate, Bulk, Slab, AdsorbateSlabConfig
import numpy as np
import torch
import ast
import glob
from ase.io import read
from tools import SiteAnalyzer
from utils import *
import warnings
warnings.filterwarnings("ignore")
from secret_keys import openapi_key, anthropic_key, deepseek_key
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OPENAI_API_KEY"] = openapi_key
os.environ['ANTHROPIC_API_KEY'] = anthropic_key
os.environ['DEEPSEEK_API_KEY'] = deepseek_key

class AdaptReasoningParser(BaseModel):
    """Information gathering plan"""

    other: Optional[str] = Field(description="other information about the adsorbate-catalyst system")

    adapted_prompts: List[str] = Field(
        description="Adapted and rephrased prompts to better identify the information required to solve the task"
    )
    preamble: Optional[str] = Field(
        description="preamble to reasoning modules"
    )  


class AdaptSolutionParser(BaseModel):
    """Information gathering plan"""

    # human_solution: Optional[List[str]] = Field(description="Human help in solving the problem")

    adsorption_site_type: str = Field(description="Type of adsorption site (e.g., ontop, bridge, hollow; in lower case)")
    binding_atoms_in_adsorbate: List[str] = Field(description="Binding atoms in the adsorbate")
    binding_atoms_on_surface: List[str] = Field(description="Binding atoms on the surface")
    number_of_binding_atoms: int = Field(description="Number of binding atoms on the surface")
    orientation_of_adsorbate: str = Field(description="Orientation of the adsorbate (e.g., end-on, side-on)")
    reasoning: str = Field(description="Reasoning for the derived configuration")
    text: str = Field(description="Textual description of the derived configuration")

class AdaptCriticParser(BaseModel):
    """Information gathering plan"""
    human_solution: Optional[List[str]] = Field(description="Human help in solving the problem")
    solution: int = Field(description="1 if the observation is correct, otherwise 0")

class AdaptIndexParser(BaseModel):
    """Plan for gathering information about binding atom indices."""
    
    human_solution: Optional[List[str]] = Field(description="Human-provided help in solving the problem")

    solution: List[int] = Field(
        description="Indices of the binding atoms in the adsorbate (0-based indexing)"
    )
    

def info_reasoning_adapter(model, parser=AdaptReasoningParser):
    information_gathering_adapt_prompt = PromptTemplate(
        input_variables=["observations", "reasoning"], 
        template=(
            "You are an expert in catalyst and surface chemistry. Based on the given adsorbate and catalyst surface, "
            "observations: {observations}\n"
            "Your task is to rephrase, rewrite, and reorder each reasoning module to better identify the information "
            "needed to derive the most stable adsorption site and configuration for adsorption energy identification. "
            "Additionally, enhance the reasoning with relevant details to determine which adsorption site and configuration "
            "shows the lowest energy for the given adsorbate and catalytic surface.\n"
            "Reasoning Modules: {reasoning}."
        )
    )
    adapter = information_gathering_adapt_prompt | (model).with_structured_output(parser)
    return adapter

def solution_planner(model, parser=AdaptSolutionParser):
    solution_planner_prompt = PromptTemplate(
        input_variables=["observations", "adapter_solution_reasoning"],
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "Your task is to find the most stable adsorption configuration of an adsorbate on the catalytic surface, "
            "including adsorption site type (ontop, bridge, hollow), binding atoms in the adsorbate and surface, their numbers, and the orientation of adsorbate (side-on, end-on, etc). "
            "Given the system: {observations}, you must operationalize "
            "the reasoning modules {adapter_solution_reasoning} to derive the most stable configuration for adsorption energy identification.\n"
            "You need to provide the most stable adsorption site & configuration with the adsorption site type, binding atoms "
            "in the adsorbate and surface, the number of those binding atoms, and the connection of those binding atoms.\n"
            "NOTE: The adsorption site can be surrounded by atoms of the same element or a combination of different elements. Avoid generating binding surface atoms by merely listing all atom types. \n"
            "Instead, determine the binding surface atoms based on the actual atomic arrangement of the surface.\n" 
            "Ensure the derived configuration is very specific and not semantically repetitive, and provide a rationale.\n"
            "Note: Do not produce invalid content. Do not repeat the same or semantically similar configuration. Stick to the given adsorbate and catalyst surface."
        )
    )

    recon = solution_planner_prompt | model.with_structured_output(parser)
    return recon

def solution_reviewer(model, parser=AdaptSolutionParser):#AdaptSolutionParser): #adapt_solution_parser):
    solution_planner_prompt = PromptTemplate(
        input_variables=["initial_configuration", "relaxed_configuration", "adapter_solution_reasoning"],
        template=(
            "You are an expert in catalysis and surface chemistry.\n"
            "Your task is to update the most stable adsorption configuration of an adsorbate on a catalytic surface.\n"
            "This includes determining the adsorption site type (on-top, bridge, hollow), identifying the binding atoms in both the adsorbate and the surface, specifying the number of binding atoms, and describing the orientation of the adsorbate (side-on, end-on, etc.).\n"
            "I have already obtained a stable relaxed configuration that shows lower energy than the initial guess of the configuration:\n"
            "Initial configuration: {initial_configuration}.\n"
            "Relaxed configuration: {relaxed_configuration}.\n"
            "You must utilize the reasoning modules {adapter_solution_reasoning} to derive a more stable configuration, referring to the initial and relaxed configurations.\n"
            "Note: Do not simply follow the relaxed configuration; instead, critically analyze and reason to derive the most stable configuration.\n"
            "You need to provide the most stable adsorption site and configuration, including the adsorption site type, the binding atoms in the adsorbate and surface, the number of those binding atoms, and the connections between those binding atoms.\n"
            "Ensure the derived configuration is very specific and not semantically repetitive, and provide a rationale.\n"
            "Note: Do not produce invalid content. Do not repeat the same or semantically similar configuration. Stick to the given adsorbate and catalyst surface."
        )
    )
    recon = solution_planner_prompt | model.with_structured_output(parser)
    return recon

def structure_analyzer(model, parser=AdaptSolutionParser):
    solution_planner_prompt = PromptTemplate(
        input_variables=["observations", "binding_information"],
        template=(
            "You are an expert in catalysis and surface chemistry.\n"
            "Your task is to convert the given adsorption configuration information into a text description.\n"
            "Given adsorbate-catalyst system: {observations}\n"
            "Binding Information: {binding_information}\n"
            "The binding information is a dictionary containing the binding atoms in the adsorbate and surface, their indices, and the binding positions.\n"
            "Provide a simplified description of the adsorption configuration based on the binding information.\n"
            "Ensure the description is clear and concise.\n"
            "In the output text description, you don't need to include the specific indices."
        )
    )
    recon = solution_planner_prompt | model.with_structured_output(parser)
    return recon



def surface_critic(model, parser=AdaptCriticParser):
    site_type_prompt = PromptTemplate(
        input_variables=["observations", "adsorption_site_type", "binding_atoms_on_surface","knowledge"],
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "Observations: {observations}\n"
            "Adsorption Site Type: {adsorption_site_type}\n"
            "Binding Atoms on Surface: {binding_atoms_on_surface}\n"
            "Knowledge: {knowledge}\n"
            "Determine whether the site type matches the number of binding surface atoms.\n"
            "If the site type matches, return 1; otherwise, return 0.\n"
        )
    )
    adapter = site_type_prompt | model.with_structured_output(parser)
    return adapter


def adsorbate_critic(model, parser=AdaptCriticParser):
    orientation_prompt = PromptTemplate(
        input_variables=["observations", "binding_atoms_in_adsorbate",  "orientation_of_adsorbate","knowledge"],
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "Observation: {observations}\n"
            "Binding Atoms in Adsorbate: {binding_atoms_in_adsorbate}\n"
            "Orientation: {orientation_of_adsorbate}\n"
            "Knowledge: {knowledge}\n"
            "Determine whether the orientation matches the binding atoms in the adsorbate.\n"
            "If the orientation fully matches, return 1; otherwise, return 0.\n"
        )
    )
    adapter = orientation_prompt | model.with_structured_output(parser)
    return adapter

def binding_indexer(model, parser=AdaptIndexParser):
    prompt_template = PromptTemplate(
        input_variables=["observations", "atomic_numbers"],
        template=(
            "You are an expert in catalyst and surface chemistry. Based on the given description of the adsorption configuration: \n"
            "Observations: {observations}\n"
            "Atomic numbers of atoms in the adsorbate: {atomic_numbers}\n"
            "Your task is to derive the indices of the binding atoms in the adsorbate. "
            "Provide the answers for the following questions (only answers, do not include the questions in the output):\n"
            "1. What are the atom indices of the adsorbate that bind to the site? (Answer: list of indices)\n"
            "Please stick to the provided answer form and keep it concise.\n"
            "Note: The indices should be 0-based."
        )
    )
    adapter = prompt_template | (model).with_structured_output(parser)
    return adapter


def singlerun_adsorb_aigent(config):
    system_info = config['system_info']
    agent_settings = config['agent_settings']
    paths = config['paths']
    metadata_path = paths['metadata_path']
    question_path = paths['question_path']
    knowledge_path = paths['knowledge_path']
    bulk_db_path = paths['bulk_db_path']
    ads_db_path = paths['ads_db_path']
    if agent_settings['provider'] == "openai":
        llm_model = ChatOpenAI(model=agent_settings['version'])
    elif agent_settings['provider'] == "anthropic":
        llm_model = ChatAnthropic(model=agent_settings['version'])
    elif agent_settings['provider'] == "deepseek":
        llm_model = ChatDeepSeek(model=agent_settings['version']) 
    gnn_model = agent_settings['gnn_model']
    critic_activate = agent_settings['critic_activate']
    mode = agent_settings['mode']
    init_multiplier = agent_settings['init_multiplier']

    # Derive the initial input prompt from system_id
    observations = derive_input_prompt(system_info, metadata_path)
    print("Input Prompt:", observations)
    reasoning_questions=load_text_file(question_path)
    knowledge_statements=load_text_file(knowledge_path)
    num_site = int(system_info["num_site"]*init_multiplier)
    random_ratio = agent_settings['random_ratio']
    save_dir = setup_save_path(config, duplicate=False)
    #########################################################################
    # if save_dir already exists, skip this config
    # check whether save_dir + /traj/*.traj exists
    traj_dir = os.path.join(save_dir, "traj")
    if os.path.exists(traj_dir):
        traj_files = glob.glob(traj_dir + '/*.traj')
        if len(traj_files) > 0:
            print(f"Skip: {config['config_name']} already exists")
            return None
    #########################################################################
    # Reasoning step
    print("Reasoning step...")
    reasoning_adapter = info_reasoning_adapter(model=llm_model)
    if agent_settings["provider"] == "openai":
        reasoning_result = reasoning_adapter.invoke({
            "observations": observations,
            "reasoning": reasoning_questions,
        })
    elif agent_settings["provider"] == "anthropic":
        reasoning_result = reasoning_adapter.invoke({
        "observations": observations,
        "reasoning": reasoning_questions,
    }, max_tokens=1024)
    elif agent_settings["provider"] == "deepseek":
        reasoning_result = reasoning_adapter.invoke({
            "observations": observations,
            "reasoning": reasoning_questions,
        }, max_tokens=1024)

    # breakpoint()
    surface_critic_valid = False
    adsorbate_critic_valid = False
    critic_loop_count1 = 0
    while not (surface_critic_valid and adsorbate_critic_valid):
        # Solution step
        print("Solution step...")
        solution_adapter = solution_planner(model=llm_model)
        solution_result = solution_adapter.invoke({
            "observations": observations,
            "adapter_solution_reasoning": reasoning_result.adapted_prompts,
        })
        if critic_activate:
            # Apply critic to evaluate the solution
            print("Critique step...")
            surface_critic_adapter = surface_critic(model=llm_model)
            surface_critic_result = surface_critic_adapter.invoke({
                "observations": observations,
                "adsorption_site_type": solution_result.adsorption_site_type,
                "binding_atoms_on_surface": solution_result.binding_atoms_on_surface,
                "knowledge": knowledge_statements,  
            })

            adsorbate_critic_adapter = adsorbate_critic(model=llm_model)
            adsorbate_critic_result = adsorbate_critic_adapter.invoke({
                "observations": observations,
                "binding_atoms_in_adsorbate": solution_result.binding_atoms_in_adsorbate,
                "orientation_of_adsorbate": solution_result.orientation_of_adsorbate,
                "knowledge": knowledge_statements,  
            })
            # Check if the critiques are valid
            surface_critic_valid = surface_critic_result.solution == 1
            adsorbate_critic_valid = adsorbate_critic_result.solution == 1
            critic_loop_count1 += 1
            print(f"critic loop count: {critic_loop_count1}")
            # Check if the critiques are valid
            # if not (surface_critic_valid and adsorbate_critic_valid):
            #     print("Critique failed. Retrying...")
            if not surface_critic_valid:
                print("Site type critique failed. Retrying...")
                print(f"Site type: {solution_result.adsorption_site_type}, Binding surface atoms: {solution_result.binding_atoms_on_surface}")
            if not adsorbate_critic_valid:
                print("Orientation critique failed. Retrying...")
                print(f"Orientation: {solution_result.orientation_of_adsorbate}, Binding atoms in adsorbate: {solution_result.binding_atoms_in_adsorbate}")
        else:
            surface_critic_valid = True
            adsorbate_critic_valid = True


    config_result = {'site_type': solution_result.adsorption_site_type,
                     'site_atoms': solution_result.binding_atoms_on_surface,
                     'num_site_atoms': solution_result.number_of_binding_atoms,
                     'ads_bind_atoms': solution_result.binding_atoms_in_adsorbate,
                     'orient': solution_result.orientation_of_adsorbate,
                     'reasoning': solution_result.reasoning,
                     }
                     #'internal_loop_count': internal_loop_count}


    # evaluate the energy
    ########### structure retriever ###########
    print("Loading adslabs...")
    if system_info.get("system_id", None) is not None:
        system_id = system_info.get("system_id", None)
        info = load_info_from_metadata(system_id, metadata_path)
    else:
        info = [
        system_info.get("bulk_id"),
        system_info.get("miller"),
        system_info.get("shift", None),
        None,  # 'top' is not provided in the fallback
        system_info.get("ads_smiles"),
        system_info.get("bulk_symbol")
    ]
    bulk_id, miller, shift, top, ads, bulk_symbol = info
    if not isinstance(miller, tuple):
        miller = ast.literal_eval(miller)
    # if num_site == 0:
    #     num_site = num
    site_type = config_result['site_type']
    site_atoms = config_result['site_atoms']
    bulk = Bulk(bulk_src_id_from_db=bulk_id, bulk_db_path=bulk_db_path)
    slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
    for slab_candidate in slabs:
        if np.isclose(slab_candidate.shift, shift, atol=0.01):
            if top is None or slab_candidate.top == top:
                slab = slab_candidate
                break
    adsorbate = Adsorbate(adsorbate_smiles_from_db=ads, adsorbate_db_path=ads_db_path)

    if mode == "llm-guided":
        index_adapter = binding_indexer(model=llm_model)
        index_result = index_adapter.invoke({
            "observations": solution_result.text,
            "atomic_numbers": adsorbate.atoms.numbers,
        })
        adsorbate.binding_indices = np.array(index_result.solution)
        # binding_atoms = [adsorbate.atoms[i] for i in index_result.solution]
        # breakpoint()
    
    #### 
    # cutoff_multiplier = 1.1
    # adslabs = []
    # while not adslabs and cutoff_multiplier <= 1.2:
    #     try:
    #         adslabs_ = AdsorbateSlabConfig(
    #             slab,
    #             adsorbate,
    #             num_sites=num_site,
    #             mode=mode,
    #             site_type=site_type,
    #             site_atoms=site_atoms,
    #             random_ratio=random_ratio,
    #             cutoff_multiplier=cutoff_multiplier
    #         )
    #         adslabs = list(adslabs_.atoms_list)
    #     except Exception:
    #         print(f"Error in creating adslabs with cutoff multiplier {cutoff_multiplier}. Retrying with a higher multiplier...")
    #         adslabs = []
    #     cutoff_multiplier += 0.02
    cutoff_multiplier = 1.1
    adslabs = []
    while not adslabs and cutoff_multiplier <= 1.3:
        try:
            adslabs_ = AdsorbateSlabConfig(
                slab,
                adsorbate,
                num_sites=num_site,
                mode=mode,
                site_type=site_type,
                site_atoms=site_atoms,
                random_ratio=random_ratio,
                cutoff_multiplier=cutoff_multiplier
            )
            adslabs = list(adslabs_.atoms_list)
        except Exception:
            print(f"Error in creating adslabs with cutoff multiplier {cutoff_multiplier}. Retrying with a higher multiplier...")
            adslabs = []
        cutoff_multiplier += 0.05
    
    # try:
    #     adslabs_ = AdsorbateSlabConfig(slab, adsorbate, num_sites=num_site, mode=mode, site_type=site_type, site_atoms=site_atoms, random_ratio=random_ratio)
    #     adslabs = [*adslabs_.atoms_list]
    # except:
    #     print("Error in creating adslabs. Skipping to the next system.")
    #     adslabs = []
    # if there is no adslabs, continue to the next system
    # breakpoint()
    if len(adslabs) == 0:
        print("No selected configurations even > 1.3 cutoff multiplier. Skipping to the next system.")
        #print("No selected configurations. Skipping to the next system.")
        # save the name of config as a failure id
        # save the result as a txt file
        # config_name = config['config_name'] 
        # with open(os.path.join(save_dir, f'{config_name}.txt'), 'w') as f:
        #     f.write(f"Error: No selected configurations for {config_name}")

        return None
    # while len(adslabs) == 0:

    
    ########### Geometry Optimizer ###########
    print("Relaxing adslabs...")
    traj_dir = os.path.join(save_dir, "traj")
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)
    relaxed_energies = []
    for i, adslab in enumerate(adslabs):
        save_path = os.path.join(traj_dir, f"config_{i}.traj")
        with torch.no_grad():
            adslab = relax_adslab(adslab, gnn_model, save_path)
        relaxed_energies.append(adslab.get_potential_energy())
        # breakpoint()

        # print('Cuda memory cleared after relaxation of config', i)
        torch.cuda.empty_cache()  # clear cuda memory after each relaxation
        torch.cuda.ipc_collect()
    min_energy = np.min(relaxed_energies)
    min_idx = np.argmin(relaxed_energies)
    ###########################################

    # Convert to dictionary
    result_dict = {'system': info}
    result_dict['initial_solution'] = config_result
    result_dict['min_energy'] = min_energy
    result_dict['min_idx'] = min_idx
    result_dict['critic_loop_count'] = critic_loop_count1
    result_dict['config_no_count'] = i+1
    result_dict['cutoff_multiplier'] = cutoff_multiplier

    print("Result:", result_dict)
    save_result(result_dict, config, save_dir)
    return result_dict


def multirun_adsorb_aigent(setting_config):
    #breakpoint()
    agent_settings = setting_config['agent_settings']
    paths = setting_config['paths']
    system_path = paths['system_dir']
    system_config_files = glob.glob(system_path + '/*.yaml')
    system_config_files.sort()

    for i, config_file in enumerate(system_config_files):
        config_name = os.path.basename(config_file)
        config_name = config_name.split('.')[0]
        
        config = load_config(config_file)
        config['config_name'] = config_name

        # combine agent_settings, paths, and system_info
        config['agent_settings'] = agent_settings
        config['paths'] = paths
        
        singlerun_adsorb_aigent(config)
        ##### clear cuda memory #####

        # print(f"Clearing GPU memory after {i} iterations...")
        torch.cuda.empty_cache()

    print('============ Completed! ============')




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Config file path')
    parser.add_argument('--path', type=str, metavar='CONFIG_FILE', 
                        help='Path to configuration file', default='config/adsorb_agent.yaml')
    args = parser.parse_args()

    config = load_config(args.path)
    multirun_adsorb_aigent(config)

    