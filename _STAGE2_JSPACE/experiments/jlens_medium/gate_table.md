# P0-3 validation gate (Medium): J-lens vs logit-lens, side by side

Lens: `jlens_gpt2_medium_100.pt` (100 prompts). Layers [2, 5, 8, 10, 12, 15, 18, 21], positions -2/-1, top-5.

## runbook_boot

`Fact: The currency used in the country shaped like a boot is`

**position -2** (token `' boot'`) — model final top-5: [' is', 'leg', '.', ' has', 'y']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[' boot', ' Boot', ' booted', ' boots', 'Boot']` | `['strap', 'stra', 'loader', 'y', '-']` |
| 5 | `[' boot', 'Boot', 'boot', ' Boot', ' booted']` | `['strap', 'stra', 'y', ' heel', 'loader']` |
| 8 | `[' boot', 'strap', ' booted', ' todd', 'boot']` | `['strap', 'stra', '-', 'y', ' and']` |
| 10 | `['strap', ' boot', ' boots', 'boot', ' booted']` | `['strap', 'stra', 'y', '-', ' and']` |
| 12 | `[' boot', ' boots', 'strap', ' booted', ' Boot']` | `['strap', 'stra', '-', 'y', ' and']` |
| 15 | `['strap', ' booted', ' boots', ' boot', ' amput']` | `['strap', 'stra', 'y', '-', ' and']` |
| 18 | `['strap', 'legged', ' nodd', ' heel', 'hoe']` | `['strap', ' is', ',', ' and', 'y']` |
| 21 | `['strap', 'legged', 'lace', 'leg', ' heel']` | `['strap', ' is', 'leg', ',', ' has']` |

**position -1** (token `' is'`) — model final top-5: [' called', ' not', ' the', ' actually', ' a']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[',', ' thus', '.', ';', ' in']` | `[' not', ' currently', ' often', ' now', ' in']` |
| 5 | `[' streng', ' arrang', ' trave', ' mathemat', ' tiss']` | `[' not', ' currently', ' bound', ' often', ' usually']` |
| 8 | `[' ingred', ' streng', ' cryst', ' arrang', ' nodd']` | `[' not', ' often', ' usually', ' now', ' NOT']` |
| 10 | `[' streng', ' arrang', ' indo', ' cryst', ' condem']` | `[' not', ' usually', ' often', ' now', ' also']` |
| 12 | `[' showc', ' condem', ' indo', ' horizont', ' shenan']` | `[' not', ' likely', ' often', ' probably', ' now']` |
| 15 | `[' distingu', ' shenan', ' indo', ' counterfeit', ' destro']` | `[' not', ' actually', ' known', ' also', ' commonly']` |
| 18 | `[' slang', ' counterfeit', ' nicknamed', ' mathemat', ' notor']` | `[' known', ' called', ' not', ' commonly', ' named']` |
| 21 | `[' counterfeit', ' called', ' nicknamed', ' counterfe', ' coined']` | `[' called', ' known', ' not', ' actually', ' named']` |

## runbook_mars

`The color of the planet fourth from the sun is`

**position -2** (token `' sun'`) — model final top-5: [' is', ',', '.', '\n', ' has']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[' sun', ' solar', ' Solar', ' Sun', ' Suns']` | `['lit', 'sets', 'rays', 'shine', 'flower']` |
| 5 | `[' solar', ' sun', 'sun', ' Solar', ' satell']` | `['lit', 'sets', 'flower', '-', 'shine']` |
| 8 | `[' solar', ' cryst', ' sun', ' satell', ' Solar']` | `['lit', ',', ' and', '-', ' Constant']` |
| 10 | `[' solar', ' celestial', ' sun', ' lunar', ' comet']` | `[',', 'lit', ' and', "'s", ' (']` |
| 12 | `[' solar', ' orbiting', ' phot', ' celestial', ' planet']` | `[',', ' (', ' and', "'s", '.']` |
| 15 | `[' orbiting', ' brightest', ' celestial', ' astronomers', ' solar']` | `[',', ' is', ' (', ' and', '.']` |
| 18 | `[' emits', ' orbiting', ' emitted', ' orbits', ' shines']` | `[' is', ',', ' has', ' (', ' was']` |
| 21 | `[' shines', ' emits', ' corresponds', ' determines', ' varies']` | `[' is', ' has', ',', ' shines', ' determines']` |

**position -1** (token `' is'`) — model final top-5: [' red', ' a', ' blue', ' the', ' called']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[',', '.', ' thus', ';', ' in']` | `[' not', ' currently', ' now', ' also', ' often']` |
| 5 | `[' arrang', ' mathemat', ' streng', ' cryst', ' destro']` | `[' not', ' currently', ' often', ' usually', ' likely']` |
| 8 | `[' cryst', ' streng', ' trave', ' ingred', ' nodd']` | `[' not', ' currently', ' often', ' usually', ' now']` |
| 10 | `[' horizont', ' mathemat', ' indo', ' trave', ' therefore']` | `[' not', ' currently', ' now', ' often', ' likely']` |
| 12 | `[' horizont', ' mathemat', ' therefore', ':', ' colours']` | `[' not', ' often', ' usually', ' currently', ' now']` |
| 15 | `[' fluorescent', ' colors', ' rainbow', ' hue', ' Transparency']` | `[' often', ' not', ' usually', ' known', ' currently']` |
| 18 | `[' fluorescent', ' rgb', ' violet', ' rainbow', ' RGB']` | `[' color', ' blue', ' purple', ' red', ' yellow']` |
| 21 | `[' violet', ' purple', ' fluorescent', ' blue', ' yellow']` | `[' blue', ' red', ' black', ' yellow', ' purple']` |

## runbook_eiffel

`The capital of the country where the Eiffel Tower stands is`

**position -2** (token `' stands'`) — model final top-5: [',', ' is', '.', ' was', ' has']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[' tall', ' upright', ' Tall', ' stands', ' stand']` | `[' up', ' out', ' at', ' before', ' on']` |
| 5 | `[' unden', ' tall', ' showc', ' arrang', ' horizont']` | `[' in', ' on', ' up', ' at', ' unpaid']` |
| 8 | `[' unden', ' arrang', ' tall', ' stands', ' advoc']` | `[' in', ' on', ' above', ' before', ' up']` |
| 10 | `[' stands', ' tall', ' pic', ' unden', ' statue']` | `[' in', ' on', ' above', '.', ',']` |
| 12 | `[' stands', ' sits', ' tall', ' tallest', ' statue']` | `[' in', ' on', ',', ' at', '.']` |
| 15 | `[' tallest', ' skyline', ' erected', ' proudly', ' today']` | `[',', '.', ' today', ' proudly', ' in']` |
| 18 | `[' tallest', ' perched', ' skyline', ' erected', ' proudly']` | `[' today', ',', ' is', ' proudly', '.']` |
| 21 | `[' dominates', ' sits', ' is', ' today', ' proudly']` | `[' is', ',', ' today', '.', ' was']` |

**position -1** (token `' is'`) — model final top-5: [' Paris', ' the', ' a', ' located', ' now']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[',', ' thus', '.', ';', ' then']` | `[' currently', ' not', ' often', ' now', ' still']` |
| 5 | `[' streng', ' arrang', ' neighb', ' nodd', ' mathemat']` | `[' not', ' currently', ' often', ' located', ' usually']` |
| 8 | `[' cryst', ' neighb', ' therefore', ' thus', ' streng']` | `[' often', ' currently', ' not', ' located', ' now']` |
| 10 | `[' therefore', ' thus', ' now', ',', ' instead']` | `[' not', ' currently', ' often', ' now', ' also']` |
| 12 | `[' unsurprisingly', ' also', ' town', ' likewise', ' Town']` | `[' named', ' not', ' also', ' located', ' often']` |
| 15 | `[' town', ' cities', ' Britann', ' city', ' municip']` | `[' also', ' now', ' often', ' not', ' currently']` |
| 18 | `[' municip', ' Budapest', ' Islamabad', ' Constantinople', ' Istanbul']` | `[' Paris', ' located', ' also', ' London', ' now']` |
| 21 | `[' Paris', ' Marse', ' France', ' Tunis', ' Stras']` | `[' Paris', ' France', ' Berlin', ' French', ' the']` |

## A03_neuro

`The hippocampal formation plays a critical role in`

**position -2** (token `' role'`) — model final top-5: [' in', ' for', ' during', ' throughout', ' both']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[' role', ' roles', ' Role', ' involvement', 'Role']` | `[' role', ' in', ' of', ' determining', ',']` |
| 5 | `[' arrang', ' conduc', ' traged', ' streng', ' proble']` | `[' in', ' determining', ' internationally', ' driving', ',']` |
| 8 | `[' advoc', ' streng', ' mathemat', ' enthusi', ' conduc']` | `[' in', ' determining', ',', ' to', ' globally']` |
| 10 | `[' conduc', ' mathemat', ' destro', ' facult', ' advoc']` | `[' in', ' determining', ' to', ',', ' genetic']` |
| 12 | `[' conduc', ' facilitating', ' mathemat', ' influencing', ' conserv']` | `[' in', ' determining', ' to', ' for', ',']` |
| 15 | `[' conduc', ' mathemat', ' embry', ' conserv', ' geop']` | `[' in', ' biologically', ' throughout', ' for', ' determining']` |
| 18 | `['��', ' adolesc', ' conduc', ' embry', ' skelet']` | `[' in', ' both', ' for', ' on', ' throughout']` |
| 21 | `[' in', ' throughout', ' during', ' skelet', ' within']` | `[' in', ' for', ' during', ' within', ' throughout']` |

**position -1** (token `' in'`) — model final top-5: [' learning', ' the', ' memory', ' spatial', ' cognitive']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[',', '.', ';', ' [', ' and']` | `[' the', ',', ' in', ' conjunction', ' and']` |
| 5 | `[' streng', ' arrang', ' veter', ' conclud', ' confir']` | `[' the', ' determining', ' producing', ' this', ' generating']` |
| 8 | `[' indo', ' streng', ' initiation', ' horizont', ' initiating']` | `[' determining', ' the', ' developing', ' generating', ' this']` |
| 10 | `[' indo', ' mathemat', ' initiation', ' streng', ' facilitating']` | `[' determining', ' the', ' developing', ' maintaining', ' this']` |
| 12 | `[' mathemat', ' initiation', ' shaping', ' facilitating', ' conduc']` | `[' determining', ' the', ' maintaining', ' developing', ' generating']` |
| 15 | `[' mathemat', ' initiation', ' physiological', ' physiology', ' neuroscience']` | `[' the', ' determining', ' human', ' maintaining', ' developing']` |
| 18 | `[' cognition', ' neuronal', ' mammalian', ' cognitive', ' adolesc']` | `[' cognitive', ' neural', ' learning', ' regulating', ' spatial']` |
| 21 | `[' neuronal', ' cognition', ' hippocamp', ' synaptic', ' spatial']` | `[' learning', ' the', ' cognitive', ' memory', ' spatial']` |

## B01_napoleon

`Napoleon crossed the Alps with an army of`

**position -2** (token `' army'`) — model final top-5: [' of', ' that', ',', ' and', '.']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[' army', ' troops', ' cavalry', ' armies', ' generals']` | `[' surplus', ' marches', ' Corps', 'camp', ' Vet']` |
| 5 | `[' cavalry', ' armies', ' army', ' troops', ' generals']` | `['illary', ' and', ' IG', ' marches', ' of']` |
| 8 | `[' generals', ' cavalry', ' armies', ' army', ' troops']` | `[' of', ',', ' consisting', ' and', ' corps']` |
| 10 | `[' generals', ' cavalry', ' armies', ' infantry', ' marched']` | `[' of', ',', ' and', ' corps', ' consisting']` |
| 12 | `[' cavalry', ' besie', ' generals', ' armies', ' marched']` | `[' of', ' consisting', ',', ' and', ' comprised']` |
| 15 | `[' marching', ' marches', ' besie', ' massac', ' cavalry']` | `[' consisting', ' of', ' comprised', ' and', ' comprising']` |
| 18 | `[' besie', ' marching', ' garrison', ' numbering', ' consisting']` | `[' consisting', ' of', ' comprised', ' comprising', ' composed']` |
| 21 | `[' numbering', ' consisting', ' comprised', ' comprising', ' garrison']` | `[' of', ' consisting', ' that', ' composed', ' numbering']` |

**position -1** (token `' of'`) — model final top-5: [' 100', ' about', ' 1', ' 200', ' 300']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[',', ';', '.]', '.', ' [']` | `[' the', ' course', ' which', ' this', ' our']` |
| 5 | `[' mathemat', ' streng', ' surpr', ' challeng', ' destro']` | `[' the', ' varying', ' equal', ' white', ' either']` |
| 8 | `[' destro', ' mathemat', ' cannons', ' desper', ' cryst']` | `[' white', ' black', ' armed', ' foreign', ' around']` |
| 10 | `[' destro', ' mathemat', ' juven', ' fiercely', ' warriors']` | `[' gu', ' foreign', ' armed', ' twenty', ' around']` |
| 12 | `[' destro', ' mathemat', ' mercenaries', ' laborers', ' warriors']` | `[' gu', ' six', ' seven', ' four', ' twenty']` |
| 15 | `[' destro', ' mercenaries', ' peasants', ' mathemat', ' helicop']` | `[' thousands', ' seven', ' 200', ' six', ' twenty']` |
| 18 | `[' mercenaries', ' troops', ' helicop', ' peasants', ' assassins']` | `[' thousands', ' 300', ' hundreds', ' 200', ' 500']` |
| 21 | `[' troops', ' mercenaries', ' soldiers', ' cavalry', ' infantry']` | `[' 300', ' troops', ' soldiers', ' 200', ' 100']` |

## C01_jack_jill

`Jack and Jill went up the hill to`

**position -2** (token `' hill'`) — model final top-5: [' to', ',', ' and', '.', ' from']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[' hill', ' hills', ' Hill', 'Hill', 'hill']` | `['top', ' to', ' Elder', 'stead', ' of']` |
| 5 | `[' hill', 'Hill', 'hill', ' Hill', ' hills']` | `['top', 'side', ' to', 'market', ' with']` |
| 8 | `[' hill', ' hills', ' Hill', ' elevation', ' valley']` | `['side', ',', ' and', ' to', ' on']` |
| 10 | `[' hill', ' hills', ' uphill', ' slope', ' slopes']` | `['side', ',', ' and', ' to', ' on']` |
| 12 | `[' hill', ' slopes', ' slope', ' hills', ' valley']` | `['side', ' to', ' and', ',', ' on']` |
| 15 | `[' slopes', ' slope', ' hill', ' uphill', ' hills']` | `[' to', ',', ' toward', ' and', ' in']` |
| 18 | `[' slopes', ' uphill', ' stairs', ' skiing', ' hills']` | `[',', ' to', ' in', ' and', '.']` |
| 21 | `[' overlooking', ' toward', ' towards', ' onto', ' uphill']` | `[' from', ' to', ',', ' and', '.']` |

**position -1** (token `' to'`) — model final top-5: [' the', ' get', ' a', ' see', ' meet']

| layer | J-lens top-5 | logit-lens top-5 |
|---|---|---|
| 2 | `[',', ';', '.', ' in', ' [']` | `[' be', ' make', ' the', ' get', ' meet']` |
| 5 | `[' confir', ' mathemat', ' destro', ' rul', ' challeng']` | `[' find', ' make', ' the', ' where', ' meet']` |
| 8 | `[' confir', ' rul', ' ingred', ' veter', ' destro']` | `[' meet', ' make', ' be', ' find', ' where']` |
| 10 | `[' horizont', ' mathemat', ' neighb', ' veter', ' destro']` | `[' meet', ' make', ' get', ' where', ' be']` |
| 12 | `[' horizont', ' their', ' scouts', ' they', ' mum']` | `[' meet', ' make', ' get', ' try', ' the']` |
| 15 | `[' horizont', ' farmland', ' helicop', ' scouts', ' scout']` | `[' meet', ' get', ' make', ' the', ' try']` |
| 18 | `[' horizont', ' assassinate', ' glim', ' retrieve', ' farmland']` | `[' meet', ' the', ' get', ' pick', ' try']` |
| 21 | `[' retrieve', ' investigate', ' meet', ' assassinate', ' celebrate']` | `[' the', ' get', ' meet', ' see', ' find']` |
