# Test d'ascolto — protocollo

Obiettivo: capire se l'audio augmentato **suona credibile** e se **si sente la
tecnica giusta**, e confrontare i motori **legacy (v1)**, **world_v2** e
**psola**. Tempo totale: circa 1 h 15 min.

## Prima di iniziare
1. Avvia l'app sul server: `python -m app.main` (ambiente `thesis-alt`), poi in
   VS Code pannello **PORTS** → *Forward a Port* → `7861` → apri
   http://127.0.0.1:7861.
2. **Non usare la tab ASR** finché gira il fine-tuning (la GPU è occupata).
3. **Cuffie**, stanza silenziosa, **volume fisso** per tutto il test (regolalo
   sul primo clip e non toccarlo più).
4. Fai una pausa di 5 minuti tra una parte e l'altra: l'orecchio si abitua
   agli artefatti.

## Parte 1 — Test alla cieca (tab *Listening test*), 2 × 40 clip, ~40 min

Non sai quale motore ascolti: è il dato più "oggettivo" che possiamo portare.

| Sessione | Rater ID da inserire | Clip |
|---|---|---|
| 1 | `antonello_s1` | 40 |
| 2 (dopo la pausa) | `antonello_s2` | 40 |

Ogni ID dà un ordine diverso; lo stesso ID ridà lo stesso ordine (puoi
interrompere e riprendere). Il mix è bilanciato: prese reali con tecnica,
prese di controllo, legacy, world_v2, psola (tutte le lingue).

**Naturalezza** — come suona, *indipendentemente dalla tecnica*:

| Voto | Significato |
|---|---|
| 5 | indistinguibile da un cantante vero |
| 4 | naturale, qualche piccolo dubbio |
| 3 | si sente che è elaborato, ma accettabile |
| 2 | chiaramente artificiale (metallico, robotico) |
| 1 | molto artificiale / fastidioso |

**Tecnica** — scegli quella che *senti davvero*, non quella che ti aspetti.
Se non senti nessuna tecnica particolare scegli **none (plain singing)**.
Non riascoltare più di 2 volte: conta la prima impressione.

I voti finiscono in `results/listening/ratings.csv` (salvati da soli).

## Parte 2 — Confronto guidato (tab *Augmentation lab*), ~35 min

Qui sai cosa ascolti: serve a capire **perché** un motore suona male.
Per ogni riga: scegli la frase, ascolta **Controllo → motore → Reale**, per
ogni motore in elenco (usa il preset **v1** = intensità tipica).

| # | Tecnica (cartella) | Lingua | Motori da confrontare |
|---|---|---|---|
| 1–2 | Vibrato | inglese, italiano | legacy, world_v2, psola |
| 3–4 | Glissando | inglese, italiano | legacy, world_v2, psola |
| 5–6 | Breathy | inglese, francese | legacy, world_v2 |
| 7–8 | Pharyngeal | inglese, francese | legacy, world_v2 |
| 9–10 | Mixed_Voice_and_Falsetto → Falsetto | inglese (voce maschile se c'è), italiano | legacy, world_v2, psola |
| 11–12 | Mixed_Voice_and_Falsetto → Mixed_Voice | inglese, cinese | world_v2 |

Scegli frasi di 4–8 secondi con note lunghe (la tecnica si sente meglio).

### Cosa ascoltare, per tecnica
- **Vibrato**: l'oscillazione è troppo larga o troppo veloce? Regolare come
  una sirena (robotico) o leggermente irregolare (umano)? Parte subito o con
  un piccolo ritardo come nel reale? Somiglia a un **trillo** (salto tra due
  note)? → sarebbe un errore.
- **Glissando**: la scivolata parte e **arriva intonata**? Va su o giù in modo
  plausibile? Suona come un "lamento/sirena" artificiale?
- **Breathy**: si sente aria **dentro** la voce, o un fruscio separato
  appoggiato sopra? (sappiamo che world_v2 ne mette un po' troppo poca)
- **Pharyngeal**: voce più "nasale/squillante", senza calo di intonazione?
  Salto di volume all'inizio/fine della parola? (world_v2 alza il volume di
  ~2 dB di troppo)
- **Falsetto**: voce più leggera/flautata? Il salto di nota è credibile o
  suona "chipmunk" (timbro da cartone animato)?
- **Mixed voice**: effetto sottile, è normale sentirlo poco; segna se non
  senti differenza dal controllo.
- **Sempre**: **clic o salti ai bordi** delle parti modificate? Il resto della
  frase deve suonare identico al controllo.

### Come annotare
Compila `results/listening/manual_notes.csv` (una riga per motore ascoltato):

| Colonna | Cosa scrivere |
|---|---|
| `n` | numero della riga dell'elenco sopra (1–12) |
| `language`, `singer`, `song`, `phrase` | come nel selettore dell'app |
| `technique`, `engine`, `preset` | es. `vibrato`, `world_v2`, `v1` |
| `naturalness` | 1–5 (scala sopra) |
| `technique_credible` | 1–5: quanto somiglia alla presa **reale** |
| `artefacts` | codici separati da `;` (vedi sotto), vuoto se nessuno |
| `notes` | una frase libera |

Codici artefatti:

| Codice | Significato |
|---|---|
| `M` | metallico / ronzante |
| `R` | robotico / modulazione troppo regolare |
| `C` | clic o salto ai bordi |
| `P` | intonazione sbagliata (stonato, direzione sbagliata) |
| `T` | timbro cambiato (altra persona, chipmunk) |
| `N` | rumore / fruscio staccato dalla voce |
| `L` | salto di volume |
| `E` | effetto esagerato |
| `I` | effetto impercettibile |

Esempio:
```
n,language,singer,song,phrase,technique,engine,preset,naturalness,technique_credible,artefacts,notes
1,english,EN-Alto-1,Yesterday,0003,vibrato,legacy,v1,2,2,R;E,vibrato larghissimo e regolare
1,english,EN-Alto-1,Yesterday,0003,vibrato,psola,v1,4,4,,quasi come il reale
```

## Dopo
Dimmi che hai finito: calcolo i risultati per motore e tecnica (naturalezza
media, tecnica riconosciuta oltre il caso, confronto con le prese reali) e li
metto nelle slide per lunedì.
