# Credit-Card-Fraud-Detection-Data-Sciene-Project
Projekt für die Klassifizierung von Kreditkartenbetrug aus dem bereitgestellten Datensatz.

## Einrichtung der Umgebung

```bash
conda env create -f environment.yml
conda activate fraud-detection
```

Wichtige Abhängigkeiten: Python 3.12, pandas 2.2, scikit-learn 1.6, imbalanced-learn 0.12, TensorFlow 2.20.0 (per pip installiert), plotly 5.24, seaborn 0.13.

## Notebook starten

```bash
jupyter notebook notebook.ipynb
```

Das Notebook verwendet den `fraud-detection` Conda-Umgebungs-Kernel (`python3`).

## Daten

- `Data/fraudTrain.csv` (~1,3 Mio. Zeilen, 351 MB) und `Data/fraudTest.csv` (~556 Tsd. Zeilen, 150 MB) sind lokale CSV-Dateien im Ordner `Data/`
- Zielspalte: `is_fraud` (binär: 0 = legitim, 1 = Betrug)
- 23 Spalten, darunter Transaktionsbetrag (`amt`), Händlerkategorie `category`, geografische Koordinaten (`lat`/`long`, `merch_lat`/`merch_long`), Zeitstempel (`trans_date_trans_time`, `unix_time`) und demografische Merkmale
- Stark unausgewogen: Betrugsfälle machen ~0,5 % aller Transaktionen aus

---

## Notebook-Schritte (Implementiert)

### Schritt 1 — Datenprofilerstellung
> 📌 **For Samuel**

**Was ist eine CSV-Datei?** Eine CSV-Datei (Comma-Separated Values, dt. kommagetrennte Werte) ist eine Tabellenkalkulation, die als reiner Text gespeichert wird — jede Zeile ist eine Transaktion, jede Spalte enthält eine Information darüber. Pythons pandas-Bibliothek lädt eine CSV-Datei in einen **DataFrame**: eine Tabelle, die man in Code aufteilen, filtern und analysieren kann.

**Form (Zeilen × Spalten):** `df.shape` gibt zwei Zahlen zurück. `(1.296.675, 22)` bedeutet 1.296.675 Transaktionszeilen und jeweils 22 Informationsspalten. Der Testdatensatz hat 555.719 Zeilen — in etwa eine 70/30-Train-Test-Aufteilung.

**Nullwerte** sind fehlende Einträge in der Tabelle. Ein Nullwert zwingt einen dazu, einen Wert vor der Modellierung zu schätzen oder aufzufüllen, was Annahmen einführt. Null Nullwerte hier bedeuten, dass die Daten vollständig sind — kein Raten erforderlich.

**Zeitliche Aufteilung:** Die Trainingsdaten enden, bevor die Testdaten beginnen. Dies spiegelt die reale Welt wider: Man trainiert mit vergangenen Transaktionen und sagt zukünftige vorher. Wenn vergangene und zukünftige Daten gemischt würden, könnte das Modell beim Training versehentlich „zukünftige" Informationen sehen und unrealistisch gute Ergebnisse melden — ein Fehler namens **Datenleck**. Die `assert`-Anweisung im Code erzwingt dies automatisch.

Lädt Train/Test-CSV-Dateien und führt strukturelle Qualitätsprüfungen durch: Form, Schema-Gleichheit, Nullwert-Anzahl, Betrugsrate pro Aufteilung und zeitliche Holdout-Validierung (alle Trainings-Zeitstempel müssen vor den Test-Zeitstempeln liegen).

### Schritt 2 — Analyse des Klassenungleichgewichts
> 📌 **For Samuel**

Quantifiziert die Seltenheit von Betrug: Anzahl, Raten und Ungleichgewichtsverhältnis pro Aufteilung.

**Wichtigste Erkenntnis:** 171 legitime Transaktionen für jeden 1 Betrugsfall — ein Ungleichgewicht von 171:1. Diese einzelne Tatsache prägt jede Metrikentscheidung im Projekt.

#### Warum Genauigkeit hier nutzlos ist
Ein Modell, das immer „kein Betrug" vorhersagt, ist **99,4 % genau** und erwischt null Kriminelle. Genauigkeit zählt alle korrekten Vorhersagen gleich und belohnt daher die faule Strategie, die seltene Klasse vollständig zu ignorieren.

#### ROC-AUC — werden Betrugsfälle höher eingestuft als legitime Fälle?
ROC-AUC beantwortet eine Frage: *„Wenn ich zufällig eine Betrugs- und eine legitime Transaktion auswähle, wie oft weist das Modell dem Betrug einen höheren Risikowert zu?"*
- Score = **1,0** → jedes Mal perfektes Ranking
- Score = **0,5** → zufällig (Münzwurf-Niveau)
- Unser Ergebnis: **0,9877** — das Modell stuft Betrug 98,8 % der Zeit über Legitim ein

ROC-AUC klingt gut, hat aber ein verstecktes Problem bei unausgewogenen Daten: Es zählt beide Klassen gleich. Mit 553.574 legitimen Fällen kann selbst ein schwaches Modell die meisten „legitim unter Betrug eingestuft"-Vergleiche allein durch Zufall richtig machen. Es sagt dir, dass das Modell gut *ordnet*, aber nicht, ob es tatsächlich *nützlich* zum Erkennen von Betrug ist.

#### PR-AUC — wie nützlich ist das Modell, wenn es etwas markiert?
PR-AUC (Precision-Recall Area Under Curve) ist die primäre Metrik hier. Es betrachtet nur die Betrugsklasse und stellt gleichzeitig zwei Fragen:

- **Präzision:** Von allen Transaktionen, die das Modell als Betrug markiert hat, welcher Anteil war echter Betrug? *(Sind meine Warnungen die Untersuchung wert?)*
- **Trefferquote:** Von allen tatsächlichen Betrugstransaktionen, welchen Anteil hat das Modell erkannt? *(Verpasse ich Kriminelle?)*

PR-AUC misst, wie gut das Modell diese zwei Fragen über *jeden möglichen Entscheidungs-Schwellenwert* ausbalanciert — nicht nur einen einzigen Grenzwert. Ein höherer PR-AUC bedeutet, dass das Modell echten Betrug besser erkennt, ohne Ermittler mit Fehlalarmen zu überhäufen.

Unser Ergebnis: **PR-AUC = 0,8835** — bedeutet 88 % der Fläche unter der Precision-Recall-Kurve sind abgedeckt. Für einen Datensatz, bei dem Betrug 0,58 % aller Transaktionen ausmacht, ist dies eine starke Leistung; ein zufälliger Klassifizierer würde ungefähr 0,006 (die Betrugsrate selbst) erreichen.

### Schritt 3 — Drift-Analyse (Train vs. Test)
> 📌 **For Samuel**

**Was ist eine „Verteilung"?** Eine Verteilung ist die Form der Werte eines Merkmals. Zum Beispiel häufen sich die meisten `amt`-Werte (Transaktionsbetrag) unter 100 $, aber einige steigen bis auf 28.000 $. Dieses Muster — wie Werte über den Bereich verteilt sind — ist die Verteilung.

**Was ist Drift?** Drift liegt vor, wenn sich diese Form zwischen Trainings- und Testdaten ändert. Stell dir vor, das Modell hat aus Wintertrainingsdaten gelernt, dass „die meisten Transaktionen unter 50 $ liegen". Wenn der Testdatensatz die Weihnachtseinkaufszeit abdeckt, in der Käufe über 200 $ häufig sind, operiert das Modell jetzt in einer anderen Welt als in der, in der es trainiert wurde. Regeln, die aus einer Periode gelernt wurden, gelten möglicherweise nicht für eine andere.

**Warum es wichtig ist:** Ein Modell, das in einer verschobenen Verteilung eingesetzt wird, macht systematisch falsche Vorhersagen — und versagt möglicherweise nicht offensichtlich. Drift-Analyse erkennt dies vor dem Einsatz. In diesem Projekt ist das einzige Merkmal mit hohem Drift `unix_time` (PSI = 11,51), was zu erwarten ist: Zeitstempel steigen natürlich im Laufe der Zeit. Alle bedeutsamen Merkmale (Beträge, Kategorien, Standorte) sind stabil.

Erkennt Verteilungsänderungen zwischen Trainings- und Einsatzperioden mithilfe von PSI (Population Stability Index) für numerische Merkmale und Anteilsverschiebungs-Tabellen für kategorische Merkmale.

| PSI | Risiko |
|---|---|
| < 0,10 | Niedrig |
| 0,10 – 0,25 | Mittel |
| ≥ 0,25 | Hoch |

### Schritt 4 — Betrugsmuster-Mining
> 📌 **For Samuel**

**Warum Muster vor dem Training erkunden?** Dieser Schritt kommt bewusst *vor* dem Aufbau eines Modells. Wenn das Modell später behauptet, `amt` sei das wichtigste Merkmal, aber dieser EDA-Schritt zeigt, dass Transaktionen über 1.000 $ 41× häufiger Betrug sind, sollten diese beiden Erkenntnisse übereinstimmen. Muster-Mining gibt dir eine Grundwahrheit, gegen die du Modellentscheidungen validieren kannst.

**Was ist Lift?** Lift misst, wie viel wahrscheinlicher Betrug in einer bestimmten Gruppe im Vergleich zum Gesamtdurchschnitt ist. Ein Lift von 41× für das Betragssegment „1.000 $+" bedeutet, dass Transaktionen in diesem Bereich 41 Mal wahrscheinlicher Betrug sind als eine zufällig ausgewählte Transaktion. Lift von 1× = durchschnittliches Risiko. Lift > 1 = erhöhtes Risiko.

**Was ist ein Interaktions-Hotspot?** Einige Merkmale sind nur in Kombination gefährlich. Die Händlerkategorie `misc_net` allein hat einen Lift von 2,5× — mäßig riskant. Aber `misc_net`-Transaktionen um Stunde 23 (23 Uhr) haben einen Lift von 45× — weit gefährlicher als jedes Merkmal allein vorhersagt. Diese kombinierten Signale sind genau das, was baumbasierte Modelle wie Random Forest erkennen sollen.

Berechnet Betrugs-Lift-Tabellen nach `category`, `hour`, `amt_band` und `category × hour`-Interaktions-Hotspots. Lift > 1 bedeutet, dass eine Gruppe riskanter als die globale Grundlinie ist. Gruppen mit hohem Lift + hohem Support werden zu Kandidatenmerkmalen.

### Schritt 5 — Datenleck- und Proxy-Risikoanalyse
> 📌 **For Samuel**

**Was ist ein Datenleck?**
Ein Datenleck liegt vor, wenn ein Merkmal Informationen enthält, die zum Zeitpunkt einer echten Vorhersage nicht verfügbar wären. Wenn beispielsweise ein Feld nur *nach* der Bestätigung von Betrug einen Wert hat, erzeugt dessen Verwendung im Training eine aufgeblähte Genauigkeit, die in der Produktion verschwindet.

Jede Spalte wird mit einem Risikoniveau und einer empfohlenen Aktion klassifiziert:

| Risiko | Aktion | Bedeutung | Beispielspalten |
|---|---|---|---|
| Hoch | `drop_now` | Direkter Bezeichner — einzigartig pro Transaktion, nutzlos als Signal, verursacht nur Datenleck | `cc_num`, `trans_num`, `first`, `last`, `street` |
| Hoch | `engineer_then_drop_raw` | Rohe Form kodiert Sequenz-/Zeitartefakte; sichere Signale zuerst extrahieren | `trans_date_trans_time`, `unix_time` |
| Mittel | `keep_with_monitor` | Persönlich verknüpfte oder Standort-Proxys; erlaubt, aber auf Bias/Drift überwachen | `dob`, `zip`, `city`, `state`, `job` |
| Niedrig | `keep` | Kein größeres Datenleck- oder Proxy-Risiko | `amt`, `category`, `merchant` |

**Ausgabe dieses Schritts:** ein `risk_df`-DataFrame mit einer Zeile pro Merkmal, das `risk_level`, `reason` und `recommendation` auflistet. Druckt auch Bezeichner-Überschneidungszählungen zwischen Train und Test (eine Null-Überschneidung bei `trans_num` bestätigt keine Datenvermischung).

---

### Schritt 6 — Feature-Strategie-Entscheidung
> 📌 **For Samuel**

Dieser Schritt übersetzt die Risikobewertung aus Schritt 5 in eine **konkrete, benannte Richtlinie**, die alle nachgelagerten Zellen teilen. Anstatt dass jede Zelle ihre eigenen Entscheidungen über Entfernen/Behalten trifft, wird hier ein maßgeblicher Satz von Konstanten definiert.

**Schlüsselvariablen (verwendet von Schritten 7 und 8):**

| Variable | Typ | Zweck |
|---|---|---|
| `DROP_NOW_COLS` | Liste | Sofort zu entfernende Spalten, werden nie verwendet |
| `DERIVED_FEATURE_MAP` | Dict | Ordnet jede rohe Spalte den neuen Merkmalen zu, die sie erzeugt |
| `KEEP_RAW_COLS` | Liste | Spalten, die nach Kodierung/Skalierung direkt in das Modell eingehen |
| `CATEGORICAL_FEATURES` | Liste | Untergruppe der Beibehaltungs-Spalten, die Label-Encoding benötigen |
| `NUMERIC_FEATURES_AFTER_ENGINEERING` | Liste | Finale numerische Merkmalsnamen nach allen Ableitungen |
| `RAW_DROP_AFTER_ENGINEERING` | Liste | Alle rohen Spalten, die nach Fertigstellung abgeleiteter Merkmale entfernt werden |
| `AGE_REFERENCE_DATE` | Zeitstempel | Festes Datum zur konsistenten Berechnung des Kundenalters |

**Entscheidungsübersicht:**

```
Rohe Spalte             Aktion                  Ergebnis
───────────────────────────────────────────────────────────────────
cc_num, first, last,
street, trans_num,   →  drop_now             →  vollständig entfernt
zip, city, state

trans_date_trans_time →  engineer_then_drop   →  hour, day_of_week, month
dob                  →  engineer_then_drop   →  age
lat/long,
merch_lat/long       →  engineer_then_drop   →  distance_km
unix_time            →  drop (kein Signal)   →  entfernt

merchant             →  keep_raw             →  label-kodierte Ganzzahl
category             →  keep_raw             →  label-kodierte Ganzzahl
gender               →  keep_raw             →  label-kodierte Ganzzahl
job                  →  keep_raw             →  label-kodierte Ganzzahl
amt                  →  keep_raw             →  numerisch, skaliert
city_pop             →  keep_raw             →  numerisch, skaliert
```

**Ausgabe dieses Schritts:** eine `feature_strategy_df`-Tabelle für die Überprüfung sowie die downstream verwendeten Richtlinienkonstanten.

---

### Schritt 7 — Feature Engineering
> 📌 **For Samuel**

**Funktion:** `engineer_features(df)` — nimmt einen rohen DataFrame und gibt eine transformierte Kopie zurück. Wird identisch auf Train und Test angewendet, um dasselbe Schema zu garantieren.

#### Teilschritt A: Neue Merkmale ableiten

**Zeitmerkmale** (aus `trans_date_trans_time`):

| Neues Merkmal | Was es erfasst | Betrugsrelevanz |
|---|---|---|
| `hour` | Tagesstunde (0–23) | Betrug häuft sich nachts (Schritt 4 Hotspots haben dies bestätigt) |
| `day_of_week` | Wochentag (0=Mo, 6=So) | Wochendmuster unterscheiden sich von Wochentagen |
| `month` | Monat des Jahres (1–12) | Saisonale Betrugsverschiebungen (Feiertagszeiten) |

**Alter** (aus `dob`):
```
age = (AGE_REFERENCE_DATE − date_of_birth).days / 365.25
```
Verwendet ein festes Referenzdatum (`2020-06-21`), damit das Alter über alle Zeilen konsistent ist, unabhängig davon, wann das Notebook ausgeführt wird.

**Geografische Distanz** (aus `lat`, `long`, `merch_lat`, `merch_long`):
```
distance_km = Haversine(cardholder_location, merchant_location)
```
Die Haversine-Formel gibt die Luftlinien-Entfernung auf der Erdoberfläche in km an. Ein Karteninhaber, der in einer weit von seiner Heimatadresse entfernten Stadt einkauft, ist ein bekanntes Betrugs-Signal.

#### Teilschritt B: Rohe Spalten entfernen

Nachdem alle abgeleiteten Merkmale existieren, entfernt `RAW_DROP_AFTER_ENGINEERING` jede Quellspalte, sodass keine rohe Form die Modellierung erreicht.

**Schema vor `engineer_features()`:** 23 Spalten (rohe CSV minus Index)
**Schema nach `engineer_features()`:** 11 Spalten

```
Vorher → Nachher

trans_date_trans_time  →  hour, day_of_week, month  (3 neu, roh entfernt)
dob                    →  age                        (1 neu, roh entfernt)
lat, long,
merch_lat, merch_long  →  distance_km               (1 neu, 4 rohe entfernt)
cc_num, first, last,
street, city, state,
zip, trans_num,
unix_time              →  (entfernt, kein Ersatz)

merchant               →  merchant                   (unverändert)
category               →  category                   (unverändert)
amt                    →  amt                        (unverändert)
gender                 →  gender                     (unverändert)
city_pop               →  city_pop                   (unverändert)
job                    →  job                        (unverändert)
is_fraud               →  is_fraud                   (Ziel, unverändert)
```

---

### Schritt 7b — Kodierung und Skalierung
> 📌 **For Samuel**

Diese Zelle konvertiert den bearbeiteten DataFrame in modellbereite NumPy-Arrays.

#### Kategorische Kodierung mit `LabelEncoder`

Die vier kategorischen Spalten (`merchant`, `category`, `gender`, `job`) enthalten Zeichenketten. Modelle benötigen Zahlen.
`LabelEncoder` weist jeder einzigartigen Zeichenkette basierend auf dem Trainingssatz eine feste Ganzzahl zu:

```
category Zeichenketten  →  Ganzzahl-Codes
─────────────────────────────────────────
"grocery_pos"           →  3
"shopping_net"          →  9
"gas_transport"         →  2
...
```

**Kritische Regel — nur auf Train anpassen:**
Der Encoder lernt die Zuordnung nur aus Trainingsdaten. Wenn eine Test-Transaktion eine Kategorie hat, die im Training nicht gesehen wurde, wird sie auf `-1` (ein sicheres Unbekannt-Signal) abgebildet, anstatt einen Fehler auszulösen.

**Ausgegebene Information:** für jede Spalte die Anzahl der eindeutigen Train-Klassen und wie viele Testwerte unbekannt waren.

#### Skalierung mit `StandardScaler`

Jedes Merkmal wird so neu skaliert, dass Mittelwert ≈ 0 und Standardabweichung ≈ 1:

```
skalierter_wert = (roher_wert − train_mittelwert) / train_std
```

**Kritische Regel — nur auf Train anpassen:**
`train_mittelwert` und `train_std` werden nur aus Trainingsdaten berechnet und dann auf den Test angewendet. Die Verwendung von Test-Statistiken würde es Testdaten ermöglichen, die Vorverarbeitung zu beeinflussen — eine Form von Datenleck.

**Warum Skalierung notwendig ist:**
`amt` reicht von ~1 $ bis ~28.000 $. `hour` reicht von 0 bis 23. Ohne Skalierung behandelt Logistische Regression eine 1-Einheit-Änderung in `amt` und eine 1-Einheit-Änderung in `hour` als gleich wichtig — was sie nicht sind. Skalierung bringt alle Merkmale auf dieselbe Größenordnung, damit der Optimierer sie fair behandelt.

> Hinweis: Random Forest ist skalierungsinvariant (Bäume teilen nach Schwellenwert, nicht nach Größenordnung), daher beeinflusst Skalierung seine Ergebnisse nicht — schadet aber auch nicht.

#### Finale Ausgabe-Arrays

| Variable | Form | dtype | Inhalt |
|---|---|---|---|
| `X_train_np` | (1.296.675 × 11) | float64 | Skalierte Trainingsmerkmale |
| `X_test_np` | (555.719 × 11) | float64 | Skalierte Testmerkmale |
| `y_train_np` | (1.296.675,) | int32 | Betrugs-Labels (0 oder 1) |
| `y_test_np` | (555.719,) | int32 | Betrugs-Labels (0 oder 1) |

Diese vier Arrays sind die einzigen Eingaben für alle Modelle in Schritt 8.

### Schritt 8 — Baseline-Modellierung (Präzision-zuerst)

Trainiert zwei klassengewichtete Baseline-Modelle und vergleicht sie:

| Modell | Typ | Behandlung des Klassenungleichgewichts |
|---|---|---|
| `log_reg_balanced` | Logistische Regression | `class_weight="balanced"` |
| `rf_balanced` | Random Forest (300 Bäume) | `class_weight="balanced_subsample"` |

#### Modell-Ausgabe-Referenz

**Pro-Transaktions-Scores (`y_score_test`):**
Ein 1D NumPy-Array der Länge ~555 Tsd. Jeder Wert ist P(Betrug) ∈ [0, 1]. Werte nahe 1 sind hochkonfidente Betrugsvorhersagen. Dieses Array speist die Schwellenwert-Kalibrierung in Schritt 9.

**Vergleichstabelle (`baseline_results_df`):**

| Spalte | Was es misst |
|---|---|
| `roc_auc` | Ranking-Qualität über alle Schwellenwerte (0,5 = zufällig) |
| `pr_auc` | Fläche unter der Precision-Recall-Kurve — **primäre Auswahlmetrik** |
| `precision_at_0_5` | Von markierten Transaktionen, Anteil echter Betrug |
| `recall_at_0_5` | Von allen echten Betrugsfällen, Anteil vom Modell erkannt |
| `f1_at_0_5` | Harmonisches Mittel von Präzision und Trefferquote bei Schwellenwert 0,5 |
| `alert_count_at_0_5` | Anzahl markierter Transaktionen — operationelle Arbeitslast |
| `alert_rate_at_0_5` | Anteil aller markierten Test-Transaktionen |

**Auswahlregel:** Das Modell mit dem höchsten `pr_auc` wird als `best_model` an Schritt 9 weitergeleitet.

#### Was die Ergebnisse für unseren Betrugs-Anwendungsfall bedeuten

Das gewinnende Modell, `rf_balanced`, erzielte diese Ergebnisse auf dem zurückgehaltenen Testdatensatz (555.719 Transaktionen, 2.145 echte Betrugsfälle):

| Metrik | Wert | Bedeutung in einfacher Sprache |
|--------|-------|-------------------------------|
| ROC-AUC | 0,9877 | Stuft einen zufälligen Betrug 98,8 % der Zeit höher ein als eine legitime Transaktion |
| **PR-AUC** | **0,8835** | Primäre Metrik — starke Präzisions-Trefferquoten-Balance über alle Schwellenwerte |
| Präzision (bei 0,5) | 95,8 % | Wenn das Modell eine Warnung auslöst, hat es 19 von 20 Mal recht |
| Trefferquote (bei 0,5) | 72,7 % | Erkennt ~1.560 der 2.145 echten Betrugsfälle; verpasst ~585 |
| Tägliche Warnungen | 1.628 | Anzahl täglich markierter Transaktionen für Ermittler |

**Was das in der Praxis bedeutet:**
- Ein Betrugsermittler, der jede Warnung prüft, würde in 19 von 20 Fällen echten Betrug finden — sehr wenige verschwendete Anrufe.
- Etwa 1 von 4 echten Betrugsfällen bleibt beim Standard-Schwellenwert 0,5 unerkannt. Schritt 9 (Schwellenwert-Kalibrierung) behebt dies: Durch Absenkung des Schwellenwerts auf 0,303 erkennt das Modell 85 % der Betrugsfälle auf Kosten von mehr (aber noch handhabbaren) Fehlalarmen.
- Logistische Regression wurde getestet, erzielte aber PR-AUC = 0,145 — sie kann die nicht-linearen Betrugsmuster (z. B. „hoher Betrag + späte Nacht + Online-Kategorie = Betrug") nicht erfassen, die Random Forest durch Feature-Interaktionen natürlich handhabt.

---

## Aktueller Stand

Schritte 1–8 sind implementiert und in `notebook.ipynb` dokumentiert. Schritte 9 und 10 sind geplant (siehe unten).

---

## Schritte 9 und 10 (Team-Planungsnotizen)

### Schritt 9: Schwellenwert-Kalibrierung (Präzision-zuerst)
**Ziel**
Einen Entscheidungs-Schwellenwert für Modell-Wahrscheinlichkeits-Scores wählen, der geschäftliche Präzisionsziele erfüllt und gleichzeitig nützliche Trefferquote beibehält.

**Warum dieser Schritt existiert**
Betrugsmodelle geben Wahrscheinlichkeiten aus. Der Betrieb benötigt eine harte Entscheidungsregel (`Betrug` vs. `kein Betrug`), und diese Regel ist der Schwellenwert.

**Eingaben**
1. Bestes in Schritt 8 ausgewähltes Baseline-Modell.
2. Test-Betrugs-Scores (`predict_proba`-Ausgabe).
3. Präzision-zuerst-Ziel (hohe Präzision, kontrolliertes Warnungsvolumen).

**Kernmethode**
1. Schwellenwerte durchlaufen (zum Beispiel `0,01` bis `0,99`).
2. Präzision, Trefferquote, F1 und Warnungsrate für jeden Schwellenwert berechnen.
3. Kandidaten-Schwellenwerte auswählen, die Präzisionsziele erfüllen (Beispiel: `95 %`, `97 %`, `99 %`).
4. Trefferquoten-Verlust und Warnungsvolumen-Kompromiss über Kandidaten vergleichen.

**Ergebnisse**
1. Schwellenwert-Vergleichstabelle.
2. Empfohlener Betriebsschwellenwert.
3. Geschäftliche Interpretation: erwartetes Warnungsvolumen und erwartete Betrugserkennungsrate.

### Schritt 10: Finale Empfehlung
**Ziel**
Das Modell + Schwellenwert-Paket finalisieren, das zur Implementierung übergehen soll.

**Warum dieser Schritt existiert**
Das Team benötigt eine klare, entscheidungsfertige Empfehlung statt mehrerer teilweise bewerteter Optionen.

**Eingaben**
1. Feature-Strategie-Entscheidungen aus Schritt 6.
2. Modell-Vergleichsergebnisse aus Schritt 8.
3. Schwellenwert-Kalibrierungsergebnisse aus Schritt 9.

**Entscheidungsregel**
1. Präzision-zuerst-Anforderung priorisieren.
2. Unter Modellen, die Präzisionserwartungen erfüllen, das mit bestem Trefferquote/PR-AUC und akzeptablem Warnungsvolumen wählen.
3. Keine Datenleck-Risiko-Verletzungen aus Schritt 5 bestätigen.

**Ergebnisse**
1. Ausgewählter Modellname und Version.
2. Ausgewählter Schwellenwert und Begründung.
3. Finale Metrik-Übersicht (PR-AUC, Präzision, Trefferquote, Warnungsrate).
4. Implementierungs-Übergabenotizen (Merkmalsliste, Vorverarbeitung, Schwellenwert-Richtlinie).

### Team-Diskussionscheckliste
1. Welche Mindestpräzision muss in der Produktion garantiert werden?
2. Welches Warnungsvolumen pro Tag/Woche ist operationell akzeptabel?
3. Ist eine niedrigere Trefferquote akzeptabel, um das Präzisionsziel aufrechtzuerhalten?
4. Benötigen wir später unterschiedliche Schwellenwerte nach Transaktionssegment?

---

## Code-Review-Notizen

| Block | Risiko / Verbesserung |
|---|---|
| `warnings.filterwarnings('ignore')` | Kann wichtige Probleme während der Entwicklung verbergen; für Debugging entfernen |
| Hardcodierte CSV-Pfade in Schritt 1 | Konfigurations-/umgebungsbasierte Pfade für Portabilität hinzufügen |
| `haversine_km` | Kein Schutz für fehlende/ungültige Koordinaten |
| Festes `AGE_REFERENCE_DATE` | Kann veralten, wenn sich der Datensatz-Zeitraum ändert |
| `LabelEncoder` auf nominalen Merkmalen | Impliziert falsche Ordinal-Bedeutung für Logistische Regression; Target-Encoding für `merchant` in Betracht ziehen |
| `StandardScaler` auf label-kodierten Ganzzahlen | Harmlos, aber für Baummodelle irreführend (die skalierungsinvariant sind) |
| Scaler/Encoder nicht gespeichert | `joblib.dump()`-Aufrufe für reproduzierbare Inferenz hinzufügen |
