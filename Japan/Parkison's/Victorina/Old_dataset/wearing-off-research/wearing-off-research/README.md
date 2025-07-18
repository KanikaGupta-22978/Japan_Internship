# README

## File Structure
The following folders contain core folders needed for the wearing-off research.
* `.devcontainer`: VS Code folder for building a dev container using VS Code and Docker. Read more about this topic in the following links.
  * [Create a Dev Container](https://code.visualstudio.com/docs/devcontainers/create-dev-container)
  * [Getting started with Docker](https://docs.docker.com/get-started/)
* `data`: contains raw and processed datasets from Garmin, wearing-off labels.
  * `0raw`: contains the separated and raw datasets from Garmin and wearing-off labels.
  * `10-person`: contains the processed and combined dataset from the 10-person data collection i.e., 1st data collection.
  * `3-person`: contains the processed and combined dataset from the 3-person data collection i.e., 2nd data collection.
* `notebooks`: contains preprocessing for cleaning and combining raw datasets, processing combined dataset, and forecasting wearing-off.

These files and folders can be ignored as they are related to producing reports via code or coding using VS Code.
* `.vscode`
* `results`
* `themes`
* `.gitignore`
* `LICENSE`

## Setup
1. Download and install [VS Code](https://code.visualstudio.com/Download) and [Docker](https://docs.docker.com/).
2. In VS Code, install the core extensions.
  1. Dev Containers
  2. Docker
  3. Remote Development
  4. WSL (for Windows OS)
5. Clone this repository.
6. Open the cloned repository in VS Code. Then, click on "Reopen in Container".
If the notification below does not show or missed out, click `CTRL + P` then choose `Dev Containers: Rebuild Without Cache and Reopen in Container`.
![image](https://github.com/jnoelvictorino/abc2023/assets/45357338/f0570d3b-4ea2-4a2b-b7e5-8e2e659efdf1)



