# Adsorb-Agent  
**Autonomous Identification of Stable Adsorption Configurations via LLM Agent**  

---

## Overview  
**Adsorb-Agent** is an LLM-powered tool designed to identify the most stable adsorption configurations on catalytic surfaces. By leveraging built-in knowledge and emergent reasoning capabilities of Large Language Models, Adsorb-Agent efficiently reduces the computational cost associated with traditional exhaustive search methods while maintaining accuracy.  

This repository also includes a baseline algorithmic approach (*ocp-demo*) for direct comparison.  

⚠️ **Note:** This project is currently under construction for perfect public usage. Some features may change or be updated, and improvements are ongoing.  

---

## Getting Started  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.8+  
- Required libraries (install via `requirements.txt`):  
   ```bash
   pip install -r requirements.txt
   ```

## Running Adsorb-Agent  

To execute Adsorb-Agent, use the following command:  
```bash
python adsorb_agent.py --path adsorb_agent_config_file
```
Replace `adsorb_agent_config_file` with the path to your configuration file.

## Running OCP-Demo (Baseline Algorithmic Approaches)

For comparison purposes, you can run the baseline algorithmic approach using the following command:

```bash
python ocp_demo.py
```

## Postprocessing  

After running Adsorb-Agent or OCP-Demo, postprocessing is required to filter out anomalies and identify the most stable adsorption configuration.

- **Postprocessing Adsorb-Agent results:**  
   ```bash
   python postprocess.py --dir result_save_path
   ```
   Replace result_save_path with the directory where the Adsorb-Agent results are saved.
- **Postprocessing OCP-Demo results:**
    ```bash
    python postprocess_ocpdemo.py --path result_save_path
    ```
    Replace result_save_path with the directory where the OCP-Demo results are saved.

## Citation  

If you use **Adsorb-Agent** in your work, please cite the following:  

**BibTeX:**  
```bibtex
@misc{ock2024adsorbagent,
      title={Adsorb-Agent: Autonomous Identification of Stable Adsorption Configurations via Large Language Model Agent}, 
      author={Janghoon Ock and Tirtha Vinchurkar and Yayati Jadhav and Amir Barati Farimani},
      year={2024},
      eprint={2410.16658},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16658}, 
}
```

## Contact  

For questions, feedback, or further information, please contact:  

**Janghoon Ock**  
**Email:** [jock@andrew.cmu.edu]  


