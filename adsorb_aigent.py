from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from fairchem.data.oc.core import Adsorbate, Bulk, Slab, AdsorbateSlabConfig

import time
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS

class adapt_reasoning_parser(BaseModel):
    """Information gathering plan"""

    other: Optional[str] = Field(description="other information about the adsorbate-catalyst system")

    adapted_prompts: List[str] = Field(
        description="Adapted and rephrased prompts to better identify the information required to solve the task"
    )
    preamble: Optional[str] = Field(
        description="preamble to reasoning modules"
    )  

def info_reasoning_adapter(model, parser=adapt_reasoning_parser):
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


class adapt_solution_parser(BaseModel):
    """Information gathering plan"""

    human_solution: Optional[List[str]] = Field(description="Human help in solving the problem")

    solution: List[str] = Field(
        description="Detailed adsorption configurations likely to be the most stable configuration, including adsorption site type, binding atoms in the adsorbate and surface, the number of binding atoms"
    )


def solution_planner(model, parser=adapt_solution_parser):
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


class adapt_input_parser(BaseModel):
    """Information gathering plan"""

    human_solution: Optional[List[str]] = Field(description="Human help in solving the problem")

    solution: List[str] = Field(
    #solution: List[Dict[str, Union[str, List[str]]]] = Field(
        description="Summarize the prompt to simple list containing \
            [adsorption site type (in lower case), \
            list of binding atoms for that site, \
            list of binding atoms in the adsorbate, \
            approximate orientation configuration of adsorbate (e.g., side-on, end-on, etc. in lower case),\
            other information]"
    )

def input_summarizer(model, parser=adapt_input_parser):
    information_gathering_adapt_prompt = PromptTemplate(
        input_variables=["observations"], 
        template=(
            "You are an expert in catalyst and surface chemistry. Based on the given description on adsorption configuration, "
            "observations: {observations}\n"
            "Your task is to summarize the observation to derive the core information. "
            "The core information includes adsorption site type, binding atoms for that site, binding atoms in the adsorbate, approximate orientation of adsorbate (e.g., side-on, end-on, etc.) \n"
            "Provide answer for following questions (only answers, don't include questions into output prompt) \n" 
            "1. What is the site type? (answer: ontop, bridge, hollow, undefined, etc) \n"
            "2. What are the surface atoms on the site? (answer: list of element symobls) \n"
            "3. What are the atoms in the adsorbate that bind to the site? (answer: list of element symbols) \n"
            "4. What is the approximate orientation configuration of the adsorbate? (answer: side-on, end-on, undefined, etc) \n"
            "5. Could you describe the adsorption configuration using other descriptions? (answer: other information. eg. bond length, orientation angle, etc) \n"
            "Stick to the provided answer form and keep the answer concise and specific. \n"
            "Don't include questions into the answer."
        )
    )
    adapter = information_gathering_adapt_prompt | (model).with_structured_output(parser)
    return adapter

def derive_input_prompt(system_id, metadata_path):
    sid_to_details = pd.read_pickle(metadata_path)
    miller = sid_to_details[system_id][1]
    ads = sid_to_details[system_id][4].replace("*", "")  
    cat = sid_to_details[system_id][5]
    prompt = f"The adsorbate is {ads} and the catalyst surface is {cat} {miller}."
    return prompt

def process_reasoning_solution(system_id,
                               mode,
                               num_site, 
                               metadata_path, 
                               question_path,
                               bulk_db_path,
                               ads_db_path, 
                               llm_model,
                               save_dir):
    # Derive the initial input prompt from system_id
    observations = derive_input_prompt(system_id, metadata_path)
    reasoning_questions=load_text_file(question_path)

    # Reasoning step
    print("Reasoning step...")
    reasoning_adapter = info_reasoning_adapter(model=llm_model)
    reasoning_result = reasoning_adapter.invoke({
        "observations": observations,
        "reasoning": reasoning_questions,
    })

    # Solution step
    print("Solution step...")
    sol_adapter = solution_planner(model=llm_model)
    sol_result = sol_adapter.invoke({
        "observations": observations,
        "adapter_solution_reasoning": reasoning_result.adapted_prompts,
    })

    # Input summarization step
    print("Input summarization step...")
    input_adapter = input_summarizer(model=llm_model)
    #breakpoint()
    input_result = input_adapter.invoke({
        "observations": sol_result.solution,
    })
    config_result = convert_dict(input_result.solution)


    # evaluate the energy
    print("Loading adslabs...")
    ## when loading the adslab, the input_result should be used!
    ## this part should be updated!!
    adslabs, info = load_adslabs(system_id, mode, num_site, config_result, metadata_path, bulk_db_path, ads_db_path)
    relaxed_energies = []
    # adslabs = adslabs[:3] # need to update!!!!!
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    traj_dir = os.path.join(save_dir, "traj")
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    print("Relaxing adslabs...")
    for i, adslab in enumerate(adslabs):
        save_path = os.path.join(traj_dir, f"adslab_{i}.traj")
        # breakpoint()
        adslab = relax_adslab(adslab, save_path)
        relaxed_energies.append(adslab.get_potential_energy())

    min_energy = np.min(relaxed_energies)
    min_idx = np.argmin(relaxed_energies)

    # Convert to dictionary
    result_dict = {'system': info}
    result_dict.update(config_result)
    result_dict['full_solution'] = sol_result.solution
    result_dict['min_energy'] = min_energy
    result_dict['min_idx'] = min_idx


    # Return the result as a dictionary with an ID (replace 'some_id' with actual identifier logic if needed)
    result = {system_id: result_dict} 
    return result

def load_adslabs(system_id,
                 mode, 
                 num_site, 
                 config_result, 
                 metadata_path, 
                 bulk_db_path, 
                 ads_db_path):
    '''
    metadata: sid_to_details dictionary

    need to update the function to return AdsorbateSlabConfig object
    '''
    # breakpoint()
    metadata = pd.read_pickle(metadata_path)
    mpid = metadata[system_id][0]
    miller = metadata[system_id][1]
    shift = metadata[system_id][2]
    top = metadata[system_id][3]
    ads = metadata[system_id][4] #.replace("*", "")  
    cat = metadata[system_id][5]
    info = [mpid, miller, shift, top, ads, cat]
    site_type = config_result['site_type']
    site_atoms = config_result['site_atoms']
    
    
    bulk = Bulk(bulk_src_id_from_db=mpid, bulk_db_path=bulk_db_path)
    slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
    for s in slabs:
        if np.isclose(s.shift, shift, atol=0.01) and s.top == top:
            slab = s
            break
    adsorbate = Adsorbate(adsorbate_smiles_from_db=ads, adsorbate_db_path=ads_db_path)
    adslabs = AdsorbateSlabConfig(slab, adsorbate, num_sites=num_site, mode=mode, site_type=site_type, site_atoms=site_atoms)
    ase_atom_list = [*adslabs.atoms_list]
    return ase_atom_list, info

def relax_adslab(adslab, save_path):
    # relax the adsorbate slab
    checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/fairchem_checkpoints/')
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    adslab.calc = calc
    opt = BFGS(adslab, trajectory=save_path)
    opt.run(fmax=0.05, steps=100)
    return adslab

def load_text_file(file_path):
    """Loads a text file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

def convert_dict(input_solution, keys=['site_type', 'site_atoms', 'ads_bind_atoms', 'orient', 'others']):
    # Clean up the list (remove extra spaces)
    cleaned_list = [item.strip() for item in input_solution]

    # Check if the input_solution has less than 5 entries and only use relevant keys
    if len(cleaned_list) < len(keys):
        # Use only the first four keys
        keys = keys[:len(cleaned_list)]

    # Create the dictionary by zipping keys and cleaned list
    result_dict = dict(zip(keys, cleaned_list))

    # Convert 'site_type' and 'orient' values to lowercase
    result_dict['site_type'] = result_dict['site_type'].lower()
    result_dict['orient'] = result_dict['orient'].lower()

    return result_dict

if __name__ == '__main__':
    import pandas as pd
    import numpy as np  
    import pickle
    from secret_key import openapi_key
    import os
    os.environ["OPENAI_API_KEY"] = openapi_key
    from langchain_openai import ChatOpenAI


    # define LLM model
    llm_model = ChatOpenAI(model = "gpt-4o") #"gpt-3.5-turbo-0125")


    # define system_id
    system_id = "71_2537_62"
    num_site = 10
    mode = "llm-guided_site_heuristic_placement"
    # define paths
    question_path = "/home/hoon/llm-agent/adsorb/reasoning.txt"
    metadata_path = "/home/hoon/llm-agent/adsorb/data/processed/updated_sid_to_details.pkl"
    bulk_db_path = "/home/hoon/llm-agent/fairchem-forked/src/fairchem/data/oc/databases/pkls/bulks.pkl"
    ads_db_path = "/home/hoon/llm-agent/fairchem-forked/src/fairchem/data/oc/databases/pkls/adsorbates.pkl"
    save_dir = f"/home/hoon/llm-agent/adsorb/{system_id}-2/"

    

    # process reasoning and solution
    result = process_reasoning_solution(system_id, 
                                        mode,
                                        num_site,
                                        metadata_path, 
                                        question_path, 
                                        bulk_db_path,
                                        ads_db_path,
                                        llm_model,
                                        save_dir)
    print(result)
    # save the result

    with open(os.path.join(save_dir, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)