### Authors: 
- Jan Polišenský (xpolis04)
- Petr Pouč (xpouc01)



# Generátor snímků žilního řečiště

Cílem projektu je pomocí klasických metod nebo využitím soupeřících sítí vytvořit generátor snímků žilního řečiště prstu nebo ruky/dlaně.

Snímky nebo videozáznamy žilního řečiště je možné najít online případně je poskytne Ing. Rydlo (případnou databázi je možné i v rámci projektu rozšířit). Generátor by měl vygenerovat databázi syntetických snímků (absolutní minimum je 100 snímků). Nedílnou součástí projektu je i verifikace výsledků (tj. zda a nakolik jsou syntetické snímky žil realistické).

Odkaz na dataset [zde](https://strade.fit.vutbr.cz/data/s/ZiMQ4fMkbHjS9dS?path=%2FBIO2021-Data%2Fdata) (Pracujeme se složkami 1-5)



## Obsah repozitáře
Repozitář obsahuje jupyter notebook s implementací cGAN modelu, synteticky generované snímky, dokumentaci a další skripty. Jmenovitě pak:
- `generator.ipynb` - Jupyter notebook s implementací cGAN modelu
- `standalone_generator.py` - Skript pro před-trénování generátoru cGAN modelu, možné spustit na GPU clusteru. 
- `synth_data.zip` - 120 výsledných syntetických snímků žilního řečiště a k nim příslušné atributy ve formátu JSON, podle nichž byly generovány.
- `final_plot.ipynb` - Vizualizace výsledků.
- `dokumentace.pdf` - Dokumentace k projektu.
- `training_progress/` - Vizualizace průběhu trénování modelu.

## Instalace, spuštění
Je nutné instalovat balíčky z `requirements.txt`. Pro spuštění je nutné mít nainstalovaný `jupyter notebook` a minimálně `python 3.8`. Testováno na verzích `python 3.8.5` a `python 3.10.12`.

Projekt má vysoké nároky na pamět systému. Výsledná velikost modelů se pohybuje v jednotkách GB. Jelikož je velikost předzpracovaných dat v řádech desítek GB, je nutné mít k dispozici 40gb paměti, tedy pokud pamět systému nedostačuje, je nutné zvětšit velikost swapu. 

Projekt byl trénován na superpočítači Karoliná: [odkaz](https://www.it4i.cz/en/infrastructure/karolina)


## Zhodnocení výsledků
Jelikož bylo nutné snímky převést do černobílého spektra a zmenšit jejich velikost, přímo neodpovídají předlohám. Avšak podobnost s předzpracovanými předlohami je velká. Vizualizace výsledků je v souboru `final_plot.ipynb`.

