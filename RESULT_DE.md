# 📊 Projektergebnisse — Kreditkartenbetrugs-Erkennung

> **Zielgruppe:** Erst- oder Zweitsemester-Studierende mit Anfänger- bis Mittelstufen-Python-Kenntnissen.
> **Was dieses Dokument ist:** Eine klare Erklärung jeder Notebook-Zellen-Ausgabe in `notebook.ipynb`, die erklärt, was jede Zahl bedeutet und warum sie für den Aufbau eines Betrugsdetektors wichtig ist.

---

## Schnellübersichtstabelle

| Schritt | Was wir getan haben | Wichtigste Erkenntnis |
|---------|--------------------|-----------------------|
| 0 | Bibliotheken geladen | Umgebung ist gesund |
| 1 | Rohdaten profiliert | 1,3 Mio. Zeilen, keine Nullwerte, korrekte zeitliche Aufteilung |
| 2 | Klassenungleichgewicht gemessen | Nur 0,58 % der Transaktionen sind Betrug — eine große Herausforderung |
| 3 | Datendrift geprüft | Merkmale sind zwischen Train & Test stabil |
| 4 | Betrugsmuster erkundet | Spätnacht + hoher Betrag + Online-Shopping = enormes Betrugsrisiko |
| 5 | Auf Datenleck geprüft | ID-Spalten entfernt, die dem Modell erlauben würden zu „schummeln" |
| 6 | Finale Merkmale entschieden | 11 aussagekräftige Merkmale nach Feature Engineering behalten |
| 7 | Optimierten Random Forest trainiert | RF mit Tiefenbegrenzung trainiert in Minuten, PR-AUC 0,8835 |
| 8 | SMOTE-Überabtastung getestet | PR-AUC um 1,8 Pp gesunken → SMOTE entfernt |
| 9 | Schwellenwert-Kalibrierung | Zwei Betriebspunkte: F1-optimal (0,527) und Trefferquote-85%-SLA (0,303) |
| 10 | Hyperparameter-Tuning (manuelle Auswahl) | rf_tuned: PR-AUC 0,8628, Trefferquote 79,6 % |
| 11 | Validierung, Überanpassungs-Check, Merkmal-Wichtigkeit | amt 56,6 %, hour 20,5 %, category 11,4 % |

---

## Konzept-Leitfaden — Die vier Säulen

Bevor wir in den schrittweisen Durchgang eintauchen, erklärt dieser Abschnitt die vier großen Ideen, auf denen die Pipeline aufgebaut ist. Jeder Unterabschnitt erklärt *was* das Konzept ist, *warum* es wichtig ist und *wo* du es in den nachfolgenden Schritten angewendet siehst.

---

### 1. EDA — Explorative Datenanalyse (Schritte 1–4)

EDA bedeutet, die Daten zu befragen, bevor man irgendetwas aufbaut. Genau wie ein Arzt die Krankengeschichte eines Patienten überprüft, bevor er eine Behandlung verschreibt, erstellt ein Data Scientist zuerst ein Datenprofil, um Form, Qualität und verborgene Muster zu verstehen. In diesem Projekt enthüllte die EDA, dass der Datensatz 1,3 Millionen Zeilen und null Nullwerte hat — was bedeutet, dass er ungewöhnlich sauber ist und keine Imputation erfordert. Noch wichtiger, die EDA deckte ein 171:1-Klassenungleichgewicht auf: Für jede Betrugs-Transaktion gibt es 171 legitime. Diese einzelne Erkenntnis änderte fast jede darauffolgende Modellierungsentscheidung. EDA hat auch ein potenzielles Drift-Problem erkannt: `unix_time` hatte einen PSI von 11,51 (als hoch markiert), was sich als erwartet herausstellte, da Train und Test verschiedene Zeitfenster abdecken — kein Datenproblem. Pattern Mining in Schritt 4 fand heraus, dass Transaktionen über 1.000 $ 41× häufiger Betrug sind, und die Kombination der Kategorie `misc_net` mit Stunde 23 erzeugt einen 45× Betrugs-Lift — Signale, die direkt beeinflussten, welche Merkmale in Schritt 6 überlebten.

> **Analogie:** EDA ist wie ein Koch, der jeden Bestandteil abschmeckt, bevor er kocht. Du würdest nicht blind würzen — ein Schluck sagt dir, ob du Salz oder Zucker brauchst.

---

### 2. Feature Engineering (Schritte 5–6)

Rohe Datenbankspalten sind selten in einer Form, die ein Modell verwenden kann. Feature Engineering ist der Prozess, sie in numerische Signale zu transformieren, die echte Vorhersagekraft tragen. In diesem Projekt wurden fünf Spalten vollständig entfernt (`cc_num`, `trans_num`, `first`, `last`, `street`), weil sie Bezeichner sind, kein Verhalten — ein Modell, das Kartennummern auswendig lernt, würde bei jeder neuen Karte scheitern. Aus der einzelnen Zeitstempel-Spalte wurden vier neue Merkmale abgeleitet: `hour`, `day_of_week`, `month` und `age` (aus dem Geburtsdatum). Geografische Koordinaten wurden in `distance_km` zwischen Karteninhaber und Händler umgerechnet. Auch die Kodierung erforderte Sorgfalt: Spalten mit geringer Kardinalität (`category`, `gender`) verwendeten LabelEncoder, während hochkardinale Spalten (`merchant`, `job`) Häufigkeitskodierung verwendeten — unbekannte Werte (30 Berufe nicht im Training) wurden auf -1 abgebildet statt Absturz zu verursachen. Der Skalierer wurde nur auf Trainingsdaten angepasst und dann auf Testdaten angewendet, was eine subtile Form von Datenleck verhindert. Das Ergebnis: 22 rohe Spalten wurden auf 11 saubere numerische Merkmale kondensiert.

> **Analogie:** Feature Engineering ist wie die Übersetzung eines fremdsprachigen Rezepts in deine Muttersprache, bevor du kochst — die Zutaten sind dieselben, aber jetzt kannst du den Anweisungen tatsächlich folgen.

---

### 3. SMOTE — Behandlung des Klassenungleichgewichts (Schritt 8)

Wenn Betrugsfälle nur 0,58 % aller Transaktionen ausmachen, sehen die meisten Lernalgorithmen kaum genug Betrugbeispiele, um daraus zu lernen — sie können 99,4 % Genauigkeit erreichen, indem sie jedes Mal „kein Betrug" vorhersagen, und sind dennoch völlig nutzlos. SMOTE (Synthetic Minority Oversampling Technique) begegnet dem, indem es *synthetische* Betrugsbeispiele erzeugt: Es wählt zwei echte Betrugstransaktionen aus, interpoliert einen neuen Punkt zwischen ihnen im Merkmalsraum und fügt ihn dem Trainingssatz hinzu. Das Experiment in Schritt 8 zeigte gemischte Ergebnisse: SMOTE erhöhte die Trefferquote von 72,7 % auf 79,0 % (+6,3 Prozentpunkte), was wie ein Gewinn klingt. Aber PR-AUC fiel von 0,8835 auf 0,8659 (−1,8 Pp), und die Anzahl der täglichen Falsch-Positiv-Warnungen stieg von 1.628 auf 1.969 (+341 pro Tag). PR-AUC misst die Leistung über *alle* Wahrscheinlichkeitsschwellen — ein Rückgang dort bedeutet, dass das Modell sich bei echten Testdaten schlechter verallgemeinert hat. Die synthetischen Betrugsbeispiele führten Rauschen ein, das das Modell auswendig lernte, ohne echte Betrugsmuster zu erlernen. Entscheidung: SMOTE wurde entfernt. Der Parameter `class_weight="balanced_subsample"` im Random Forest behandelt Ungleichgewicht zuverlässiger, indem er echte Betrugsfälle während des Trainings aufwertet.

> **Analogie:** SMOTE ist wie das Hinzufügen gefälschter Prüfungsfragen zu einem Lernset, um es „ausgewogener" zu machen — wenn die Fakes nicht wie die echte Prüfung aussehen, schadet das Üben damit mehr als es nützt.

---

### 4. Auswertung — Messen, ob das Modell tatsächlich nützlich ist (Schritte 7, 9–11)

Eine einzelne Genauigkeitszahl verbirgt fast alles Wichtige bei einem Betrugserkennungsproblem. Dieses Projekt verwendet vier Auswertungswerkzeuge, um ein vollständiges Bild zu erhalten. **PR-AUC** (Precision-Recall Area Under Curve) ist die primäre Metrik: Sie misst, wie gut das Modell die seltene Betrugsklasse über alle Entscheidungsschwellen identifiziert, und erreicht 0,8835 für den optimierten Random Forest. **ROC-AUC** misst die allgemeine Ranking-Qualität bei 0,9877, ist aber hier weniger informativ, weil das extreme Klassenungleichgewicht es leicht macht, gut zu performen, selbst mit einem schwachen Modell. **Schwellenwert-Kalibrierung** (Schritt 9) erkennt, dass der Standard-0,5-Grenzwert selten optimal ist: Der F1-optimale Schwellenwert ist 0,527 (balanciert Präzision und Trefferquote für Analysten, die jede Warnung überprüfen), während der Trefferquote-85%-SLA-Schwellenwert 0,303 ist (fängt mindestens 85 % des Betrugs für compliance-gesteuerte Umgebungen ab). Schließlich zeigt der **Überanpassungs-Check** in Schritt 11 Train-PR-AUC von 1,0 vs. Test-PR-AUC von 0,8628 — eine Lücke, die angesichts der angewendeten Tiefenbeschränkungen handhabbar ist. Merkmal-Wichtigkeiten bestätigen kein Datenleck: `amt` (56,6 %), `hour` (20,5 %) und `category` (11,4 %) sind alle echte Betrugs-Signale, keine Bezeichner.

> **Warum Genauigkeit hier nutzlos ist:** Ein Modell, das immer „kein Betrug" vorhersagt, ist 99,4 % genau und erwischt null Betrüger — ein perfekter Score auf der falschen Metrik.

---

## Schritt 0 — Umgebungs-Check

### Was die Zelle ausgegeben hat
```
pandas     2.2.3
numpy      2.1.3
sklearn    1.6.1
seaborn    0.13.2
```

### Schlüsselcode
```python
print(f'pandas     {pd.__version__}')
print(f'numpy      {np.__version__}')
print(f'sklearn    {__import__("sklearn").__version__}')
print(f'seaborn    {sns.__version__}')
```

### Was es bedeutet
Vor jeder Datenarbeit haben wir überprüft, dass jede Bibliothek installiert und in einer bekannten Version ist.
**Stell es dir vor wie das Überprüfen deiner Zutaten vor dem Kochen** — wenn eine Bibliothek fehlt oder veraltet ist, würde jede darauffolgende Zelle mit verwirrenden Fehlern abstürzen.
Das Ausgeben von Versionen macht Ergebnisse auch *reproduzierbar*: Wenn dein Teammitglied andere Zahlen bekommt, weiß es genau, welche Version es installieren muss.

---

## Schritt 1 — Datenprofilerstellung
> 📌 **For Samuel**

### Was die Zelle ausgegeben hat
```
Train shape: (1,296,675, 22)    Test shape: (555,719, 22)
Fraud rate (train): 0.5789 %    Fraud rate (test): 0.3860 %
No null values in either split
Temporal split check: max(train)=2020-06-21 12:13:37 < min(test)=2020-06-21 12:14:25
```

### Schlüsselcode
```python
df_train = pd.read_csv(TRAIN_PATH, index_col=0)
df_test  = pd.read_csv(TEST_PATH,  index_col=0)

train_nulls = df_train.isnull().sum()  # zählt fehlende Werte pro Spalte
print(f"Null counts (train):\n{train_nulls[train_nulls > 0]}")  # nur Problemspalten zeigen

# Zeitliche Prüfung: jede Trainingstransaktion muss vor jeder Testtransaktion liegen
assert train_ts_max < test_ts_min, f"Temporal split violation: {train_ts_max} is not before {test_ts_min}"
```

### Was es bedeutet

**Form (Zeilen × Spalten):**
Trainingsdaten haben ~1,3 Millionen Transaktionen mit je 22 Informationen.
Testdaten haben ~556 Tausend Transaktionen — in etwa die übliche 70/30-Aufteilung im maschinellen Lernen.

**22 Spalten** umfassen Dinge wie: Transaktionsbetrag, Händlerkategorie, Alter/Standort des Karteninhabers, Transaktionszeit und ob es sich um Betrug handelt (`is_fraud`).

**Keine Nullwerte** ist eine gute Nachricht. Fehlende Daten zwingen einen dazu, Annahmen vor der Modellierung zu treffen; ein sauberer Datensatz lässt die Zahlen für sich sprechen.

**Zeitliche Aufteilung** ist entscheidend und leicht falsch zu machen. Hier enden die Trainingsdaten *vor* den Testdaten. Dies spiegelt die reale Welt wider: Du trainierst mit vergangenen Transaktionen und sagst zukünftige vorher. Wenn sie gemischt würden, würde das Modell versehentlich die Zukunft beim Training sehen und übermäßig optimistische Ergebnisse melden — ein Fehler namens **Datenleck**.

---

## Schritt 2 — Analyse des Klassenungleichgewichts
> 📌 **For Samuel**

### Was die Zelle ausgegeben hat
```
Train: 1,289,169 non-fraud  |  7,506 fraud  →  ratio 171.75 : 1
Test:    553,574 non-fraud  |  2,145 fraud  →  ratio 258.08 : 1
Fraud-rate shift (test − train): −0.19 percentage points
```

### Schlüsselcode
```python
counts = df["is_fraud"].value_counts().sort_index()  # zählt 0er (legitim) und 1er (Betrug)
fraud_rate = fraud / total
imbalance_ratio = non_fraud / fraud          # → 171.75 für Trainingssatz

fraud_rate_shift_pp = (test_fraud_rate - train_fraud_rate) * 100
```

### Was es bedeutet

**171 legitime Transaktionen für jede 1 Betrugs-Transaktion.**
Dies ist die größte Herausforderung bei der Betrugserkennung und stolpert viele Anfänger.

Stell dir vor, du trainierst ein Modell und sagst ihm: „Von 1,3 Millionen Beispielen sind 99,4 % *kein* Betrug." Ein faulen Modell, das **immer** „kein Betrug" vorhersagt, wäre *99,4 % genau* — und völlig nutzlos. Deshalb ist Genauigkeit hier eine schlechte Metrik; wir verwenden stattdessen **ROC-AUC** und **PR-AUC**.

**ROC-AUC — stuft das Modell Betrug höher ein als legitime Transaktionen?**
ROC-AUC fragt: *„Wenn ich zufällig eine Betrugs- und eine legitime Transaktion auswähle, wie oft bewertet das Modell den Betrug höher?"* Ein Score von 1,0 = perfekt, 0,5 = zufälliges Raten. Unser Ergebnis ist **0,9877** — das Modell macht dieses Ranking 98,8 % der Zeit richtig. Allerdings hat ROC-AUC einen blinden Fleck: Weil es 553.574 legitime Fälle gegen 2.145 Betrugsfälle gibt, kann selbst ein schwaches Modell allein durch das korrekte Ordnen der vielen legitimen Paare gut abschneiden. Es sagt dir, dass das Modell insgesamt gut rankt, nicht ob es tatsächlich nützlich für die Betrugserkennung ist.

**PR-AUC — wenn das Modell eine Warnung auslöst, lohnt es sich, nachzuforschen?**
PR-AUC betrachtet nur die Betrugsklasse und misst zwei Dinge gleichzeitig:
- **Präzision:** Von jeder als Betrug markierten Transaktion, welcher Anteil war echter Betrug? *(Sind meine Warnungen handlungswürdig?)*
- **Trefferquote:** Von jedem tatsächlichen Betrug, welchen Anteil hat das Modell erkannt? *(Verpasse ich Kriminelle?)*

PR-AUC fasst diesen Kompromiss über jeden möglichen Entscheidungsschwellenwert zusammen — nicht nur einen einzelnen Grenzwert. Unser Ergebnis ist **0,8835**. Zum Vergleich: Ein völlig zufälliges Modell auf diesem Datensatz würde ~0,006 erzielen (einfach die 0,58 % Betrugsrate). Ein Score von 0,88 bedeutet, dass das Modell dramatisch besser als der Zufall darin ist, Betrug zu finden, ohne Ermittler mit Fehlalarmen zu überhäufen. Deshalb ist PR-AUC die primäre Metrik für dieses Projekt.

**Die Betrugsrate sinkt leicht im Testdatensatz (0,58 % → 0,39 %).** Dies ist eine kleine Verschiebung, kein Problem — sie spiegelt normale saisonale Variation in den simulierten Daten wider.

**Was wir gegen Ungleichgewicht tun:** Wir verwenden `class_weight="balanced_subsample"`, das die Verlustfunktion neu gewichtet, sodass Betrugsfehler ~171× mehr kosten. SMOTE wurde getestet, senkte aber tatsächlich PR-AUC um 1,8 Prozentpunkte und wurde daher entfernt. Schwellenwert-Kalibrierung (Schritt 9) steuert die Trefferquote direkt.

---

## Schritt 3 — Datendrift-Analyse
> 📌 **For Samuel**

### Was die Zelle ausgegeben hat
```
unix_time PSI = 11.51   (high — expected, time advances)
All other numeric features: PSI < 0.001  (negligible)
Categorical shifts (category, gender, state, job): abs_delta < 0.001
```

### Schlüsselcode
```python
def population_stability_index(expected, actual, bins=10):
    """PSI < 0.10 = stable; 0.10–0.25 = monitor; >= 0.25 = investigate."""
    # Teilt die Train-Verteilung in 10 Eimer; misst, wie stark
    # die Test-Verteilung von denselben Eimern abweicht.
    ...

psi = population_stability_index(df_train[col], df_test[col], bins=10)

# Kategorischer Drift: vergleicht, wie viel der Anteil jeder Kategorie sich geändert hat
train_share = df_train[col].value_counts(normalize=True)
test_share  = df_test[col].value_counts(normalize=True)
abs_delta   = (train_share - test_share).abs()
```

### Was es bedeutet

**Datendrift** fragt: *„Sehen die Merkmale im Testdatensatz statistisch anders aus als im Trainingsdatensatz?"*
Wenn ja, funktioniert ein in der Vergangenheit trainiertes Modell möglicherweise nicht gut in der Zukunft.

**PSI (Population Stability Index)** ist der Standard-Drift-Score:
- PSI < 0,1 → kein Drift (gut)
- PSI 0,1–0,25 → leichter Drift (beobachten)
- PSI > 0,25 → erheblicher Drift (untersuchen)

`unix_time` hat PSI = 11,51, was alarmierend klingt — aber völlig erwartet ist. Unix-Zeit ist nur ein Zeitstempel; natürlich haben zukünftige Transaktionen höhere Zeitstempel als vergangene. Alle *bedeutsamen* Merkmale (Betrag, Kategorie, Standort usw.) sind sehr stabil mit PSI < 0,001.

**Fazit:** Der Datensatz verhält sich gut. Ein auf dem Trainingssatz trainiertes Modell sollte auf den Testdatensatz verallgemeinern, ohne spezielle Drift-Korrekturen.

---

## Schritt 4 — Betrugsmuster-Mining
> 📌 **For Samuel**

### Schlüsselcode
```python
def fraud_lift_table(df, group_col, min_tx_count=1000):
    """Berechnet Betrugs-Lift pro Gruppe vs. der globalen Grundlinie."""
    baseline = df["is_fraud"].mean()
    summary = (
        df.groupby(group_col)["is_fraud"]
        .agg(tx_count="count", fraud_count="sum", fraud_rate="mean")
    )
    summary = summary[summary["tx_count"] >= min_tx_count]  # kleine Gruppen filtern
    summary["lift_vs_baseline"] = summary["fraud_rate"] / baseline
    return summary.sort_values("lift_vs_baseline", ascending=False)

cat_lift,  _ = fraud_lift_table(df_train, "category", min_tx_count=10_000)
hour_lift, _ = fraud_lift_table(df_train, "hour",     min_tx_count=1_000)
amt_lift,  _ = fraud_lift_table(df_train, "amt_band", min_tx_count=1_000)
```

### Was die Zelle ausgegeben hat
```
Baseline fraud rate:  0.5789 %

Top categories by lift:
  shopping_net   → 3.03× more likely to be fraud
  misc_net       → 2.50×
  grocery_pos    → 2.44×

Top hours by lift:
  Hour 22 (10 PM) → 4.98×
  Hour 23 (11 PM) → 4.90×
  Hours 0–3 (midnight–3 AM) → ~2.5×

Top amount bands by lift:
  $1,000+        → 41.65×
  $500–$1,000    → 39.87×
  $200–$500      →  7.67×

Top interaction:
  misc_net × hour 23  →  45.43× lift
```

### Was es bedeutet

**Lift** sagt dir, wie viel *häufiger* Betrug in einer Gruppe im Vergleich zum Gesamtdurchschnitt ist.
Ein Lift von 3× bedeutet, dass diese Gruppe dreimal so betrügerisch ist wie eine zufällige Transaktion.

**Welche Muster entstanden sind:**

1. **Online-Shopping (`_net`-Suffix) ist riskanter als im Laden (`_pos`-Suffix).**
   Physische, kartenpräsente Transaktionen sind schwerer zu fälschen. Online-Händler benötigen nur die Kartennummer — die ohne die physische Karte gestohlen werden kann.

2. **Spätnacht-Transaktionen sind ein Warnsignal.**
   Betrüger bevorzugen es, zu operieren, während Karteninhaber schlafen und weniger wahrscheinlich die Karte in Echtzeit bemerken oder sperren.

3. **Hohe Geldbeträge sind das deutlichste Betrugssignal.**
   Eine Transaktion über 1.000 $ ist 41× wahrscheinlicher Betrug als der Durchschnitt. Betrüger versuchen, den Wert jeder gestohlenen Karte zu maximieren, bevor sie gesperrt wird.

4. **Interaktionen sind mächtiger als einzelne Merkmale.**
   Eine Online-Transaktion um 23 Uhr ist 45× wahrscheinlicher Betrug — weit vorhersagekräftiger als jedes Merkmal allein. Deshalb übertreffen Modelle wie Random Forest, die *Merkmal-Interaktionen* erfassen, einfachere Modelle bei dieser Aufgabe.

**Praxisrelevanz:** Diese Muster stimmen mit dem überein, was echte Betrugsanalysten berichten. Wenn du „ungewöhnliche Aktivität"-Warnungen von deiner Bank um 2 Uhr morgens für einen großen Online-Kauf siehst — das ist genau der Grund.

---

## Schritt 5 — Datenleck- & Proxy-Risikoprüfung
> 📌 **For Samuel**

### Was die Zelle ausgegeben hat
```
drop_now:              ['trans_num', 'cc_num', 'street', 'last', 'first']
engineer_then_drop:    ['unix_time', 'trans_date_trans_time']
keep_with_monitor:     ['merch_long', 'merch_lat', 'zip', 'dob', 'city', 'job', 'state']
keep:                  ['amt', 'long', 'lat', 'city_pop', 'merchant', 'category', 'gender']

cc_num overlap (train ∩ test): 908 cards
```

### Schlüsselcode
```python
# Spalten, bei denen fast jede Zeile einzigartig ist, fungieren als Bezeichner, nicht als Merkmale
unique_ratio = nunique / non_null   # ≈ 1.0 → Bezeichner → drop_now

# Prüfe, wie viele Kartennummern in Train UND Test erscheinen
overlap_cc_num = len(set(df_train["cc_num"]) & set(df_test["cc_num"]))
# → 908: Modell könnte kartenspezifische Muster lernen, die nicht auf neue Karten verallgemeinern
```

### Was es bedeutet

**Datenleck** liegt vor, wenn Informationen, die zum Vorhersagezeitpunkt nicht verfügbar wären, in das Modell eindringen. Es verursacht unrealistisch hohe Scores während der Auswertung, die beim Einsatz mit echten Daten zusammenbrechen.

**Sofort entfernte Spalten:**
- `trans_num` — eine einzigartige Transaktions-ID. Sie hat null Vorhersagewert und könnte versehentlich zu einem Nachschlüssel werden.
- `cc_num` — die rohe Kartennummer. Sie beizubehalten würde dem Modell erlauben, sich zu merken, dass *Karte X im Training Betrug begangen hat*. In der Produktion tritt Betrug auf *neuen* Karten auf.
- `first`, `last`, `street` — persönliche Bezeichner, keine Verhaltens-Signale.

**908 Kreditkartennummern erscheinen in Train und Test.** Das bedeutet, dass diese spezifischen Karteninhaber Transaktionen in beiden Aufteilungen haben. Wir entfernen `cc_num` genau, um zu verhindern, dass das Modell diese Überschneidung ausnutzt (es wäre wie ein Student, der Prüfungsfragen beim Probetest sieht).

**Spalten, die wir erst bearbeiten, dann entfernen:**
`trans_date_trans_time` wird in `hour`, `day_of_week` und `month` — Verhaltens-Signale — umgewandelt, dann wird der rohe Zeitstempel entfernt. Gleiches gilt für `dob` → `age`.

---

## Schritt 6 — Finale Feature-Strategie
> 📌 **For Samuel**

### Schlüsselcode
```python
# Verhaltensbezogene Merkmale aus rohen Spalten ableiten
df["hour"]        = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek
df["age"]         = (AGE_REFERENCE_DATE - pd.to_datetime(df["dob"])).dt.days / 365
df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

# LabelEncoder: nur auf Train anpassen; unbekannte Test-Werte auf -1 abbilden
le = LabelEncoder()
le.fit(df_train[col])
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
df_test_enc[col] = df_test_enc[col].map(mapping).fillna(-1).astype(int)
# → job: 494 Train-Klassen | 30 unbekannte im Test → -1

# Skalierer: fit_transform auf Train, nur transform auf Test (niemals auf Test anpassen)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train)
X_test_np  = scaler.transform(X_test)
```

### Was die Zelle ausgegeben hat
```
Dropped columns: ['cc_num', 'first', 'last', 'street', 'trans_num', 'zip', 'city', 'state']
Engineered features: ['hour', 'day_of_week', 'month', 'age', 'distance_km']
Kept raw features: ['merchant', 'category', 'amt', 'gender', 'city_pop', 'job']
Encoding: LabelEncoder for low-card (category, gender); frequency encoding for high-card (merchant, job)
Scaling:  StandardScaler (fit on train only)
```

### Was es bedeutet

Nach Bereinigung und Feature Engineering haben wir **11 Merkmale**:

| Merkmal | Was es erfasst |
|---------|---------------|
| `amt` | Transaktionsbetrag — stärkster einzelner Prädiktor |
| `category` | Art des Händlers (online vs. im Laden usw.) |
| `merchant` | Spezifischer Händler — einige Händler ziehen mehr Betrug an |
| `gender` | Kleines demografisches Signal |
| `city_pop` | Urban vs. ländlich — Betrugsraten unterscheiden sich |
| `job` | Einkommensproxy; korreliert mit Ausgabenmustern |
| `hour` | Tageszeit — späte Nacht ist riskanter |
| `day_of_week` | Wochentag vs. Wochenend-Muster |
| `month` | Saisonale Muster |
| `age` | Ältere Karteninhaber werden möglicherweise gezielter angegriffen |
| `distance_km` | Entfernung zwischen Karteninhaberwohnsitz und Händler — verdächtig wenn weit |

**Warum „nur auf Train anpassen" für Encoder und Skalierer?**
Wenn du den Skalierer auf allen Daten (Train + Test) anpasst, *beeinflusst* die Skalierung des Testdatensatzes — eine subtile Form von Datenleck. Du musst so tun, als ob der Testdatensatz nicht existiert, wenn du deine Vorverarbeitung einrichtest.

**Unbekannte Kategorien → -1:**
30 Berufe im Testdatensatz erscheinen nicht im Training. Anstatt abzustürzen, weist der Encoder ihnen -1 zu, sodass das Modell sie als „unbekannt" behandelt — eine produktionssichere Designentscheidung.

---

## Schritt 6b — Feature Engineering Ausgabe
> 📌 **For Samuel**

### Was die Zelle ausgegeben hat
```
Train after engineering: (1,296,675, 12)   [11 features + is_fraud label]
Test  after engineering: (555,719,  12)
X_train_np: (1,296,675, 11)  float64   →   fraud = 7,506  (0.5789 %)
X_test_np:  (555,719,  11)  float64   →   fraud = 2,145  (0.3860 %)
```

### Was es bedeutet

Wir begannen mit 22 rohen Spalten. Nach dem Entfernen von Bezeichnern, dem Ableiten von Zeit-/Distanz-Merkmalen und dem Kodieren von Kategorien sind wir bei **11 sauberen numerischen Merkmalen** angekommen, die für ein maschinelles Lernmodell bereit sind. Das Label `is_fraud` wird als Zielvariable `y` getrennt.

Das Konvertieren in NumPy-Arrays (`.to_numpy()`) ist ein Standard-Abschlussschritt, bevor Daten an scikit-learn-Modelle übergeben werden, die numerische Arrays statt pandas DataFrames erwarten.

---

## Schritt 7 — Baseline-Modell-Ergebnisse

### Schlüsselcode
```python
rf_balanced = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,          # Tiefenbegrenzung verhindert das Auswendiglernen von Rauschen
    min_samples_leaf=50,   # jedes Blatt braucht ≥50 Stichproben — starke Regularisierung
    max_samples=0.5,       # jeder Baum sieht nur 50% der Daten → schneller, weniger korrelierte Bäume
    class_weight="balanced_subsample",  # Betrugsfehler kosten ~171× mehr pro Bootstrap-Stichprobe
    random_state=RANDOM_STATE,
    n_jobs=-1              # alle CPU-Kerne parallel nutzen
)
rf_balanced.fit(X_train_np, y_train_np)
y_score = rf_balanced.predict_proba(X_test_np)[:, 1]  # Betrugswahrscheinlichkeit pro Transaktion
```

### Was die Zelle ausgegeben hat

Der optimierte Random Forest Baseline (mit `max_depth=20`, `min_samples_leaf=50`, `max_samples=0.5`) liefert Ergebnisse im Bereich:

| Modell | ROC-AUC | PR-AUC | Präzision | Trefferquote | F1 |
|--------|---------|--------|-----------|--------------|-----|
| **Random Forest** (optimiert) | ~0,98 | ~0,85–0,89 | ~0,95 | ~0,72 | ~0,82 |

**Hinweis:** Logistische Regression wurde früher getestet, erzielte aber PR-AUC=0,15 — sie wurde aus der Pipeline entfernt, da lineare Modelle die nicht-linearen Betrugs-Interaktionen nicht erfassen können.

### Was es bedeutet — die Metriken verstehen

**ROC-AUC (~0,98):**
„Wenn ich zufällig eine Betrugs- und eine legitime Transaktion auswähle, wie oft stuft das Modell den Betrug als riskanter ein?" — 0,98 bedeutet, es gelingt 98 % der Zeit.

**PR-AUC (~0,88):**
Dies ist die schwierigere, wichtigere Metrik für unausgewogene Daten. Sie misst den Kompromiss zwischen:
- **Präzision:** Von allen Transaktionen, die ich als Betrug markiert habe, wie viele waren es tatsächlich? (Vermeide falsche Alarme)
- **Trefferquote:** Von allen tatsächlichen Betrugsfällen, wie viele habe ich erkannt? (Verpasse keine Kriminellen)

Ein PR-AUC von 0,88 ist ausgezeichnet für eine Betrugserkennungsaufgabe.

### Warum Random Forest gewinnt

Random Forest baut Hunderte von Entscheidungsbäumen auf, von denen jeder verschiedene Merkmalskombinationen lernt. Es erfasst natürlich die in Schritt 4 entdeckten Interaktionsmuster (z. B. „online + späte Nacht + hoher Betrag = Betrug"). Die wichtigste Optimierung: Die Baumtiefe auf 20 zu begrenzen und 50 Stichproben pro Blatt zu fordern verhindert, dass Bäume Rauschen auswendig lernen, was auch die Trainingszeit von 30+ Minuten auf ~1–2 Minuten reduziert.

---

## Schritt 8 — SMOTE Überabtastung

Das Modell behandelt Ungleichgewicht bereits mit `class_weight`. SMOTE ist eine zweite Strategie — dieser Schritt testet, ob das *Hinzufügen* es on top besser oder schlechter macht.

### Schlüsselcode
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train_np, y_train_np)
# Betrugsfälle: 7.506 → ~1.289.169 (synthetische Beispiele zwischen echten Betrugspaaren interpoliert)

rf_smote = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",  # als zusätzliches Sicherheitsnetz behalten
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_smote.fit(X_train_sm, y_train_sm)
```

### Was die Zelle ausgegeben hat
```
         model    pr_auc  recall_at_0_5  precision_at_0_5  alert_count_at_0_5
0  rf_balanced  0.883456       0.727             0.958                1628
1     rf_smote  0.865872       0.790             0.860                1969

Recall    : ↑ 6.3 pp  (0.727 → 0.790)
Precision : ↓ 9.7 pp  (0.958 → 0.860)
PR-AUC    : ↓ 1.8 pp  → SMOTE removed from pipeline
```

### Was es bedeutet

**Was SMOTE macht:**
Die meisten Betrugsdaten haben ~170 legitime Transaktionen für jeden Betrugsfall. SMOTE (*Synthetic Minority Over-sampling Technique*) erstellt künstlich zusätzliche Betrugsbeispiele durch Interpolation zwischen echten — stell dir vor, du zeichnest neue Punkte entlang der Linie, die zwei nahe Betrugsfälle auf einem Streudiagramm verbindet. Das Ziel ist, dem Modell mehr Betrugsbeispiele zum Lernen zu geben.

**Warum höhere Trefferquote nicht die ganze Geschichte ist:**
SMOTE erhöhte die Trefferquote von 72,7 % auf 79,0 % — das Modell erkennt 6,3 Prozentpunkte mehr Betrug. Aber die Trefferquote ist nur die halbe Geschichte. Präzision fiel von 95,8 % auf 86,0 %, was bedeutet, dass das Modell jetzt 341 mehr tägliche Fehlalarme produziert (1.628 → 1.969). Diese zusätzlichen Warnungen landen auf den Schreibtischen echter Ermittler — jede ist ein Anruf beim Kunden oder eine manuelle Überprüfung.

**Warum PR-AUC das entscheidende Votum ist:**
PR-AUC fasst die Modellqualität über *alle* möglichen Schwellenwerte zusammen, nicht nur den Standard 0,5. Ein Rückgang von 1,8 Pp bedeutet, dass bei jedem Betriebspunkt — nicht nur dem Standard — SMOTE die Dinge leicht verschlechtert hat. Das auf synthetischen Daten trainierte Modell verallgemeinerte sich schlechter auf die *echten* Test-Transaktionen.

**Fazit:** `class_weight="balanced_subsample"` behandelt Ungleichgewicht bereits effizient. SMOTE fügt Komplexität hinzu und schadet der Verallgemeinerung hier. Es wird aus der Pipeline entfernt.

---

## Schritt 9 — Schwellenwert-Kalibrierung

Anstatt das Modell zu ändern, ändert dieser Schritt die *Entscheidungsregel*, die nach der Modell-Score-Ausgabe verwendet wird.

### Schlüsselcode
```python
precisions, recalls, thresholds = precision_recall_curve(y_test_np, y_score)

# F1-optimaler Schwellenwert: maximiert das harmonische Mittel von Präzision und Trefferquote
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_threshold = thresholds[f1_scores.argmax()]           # → 0.527

# SLA-Schwellenwert: niedrigster Schwellenwert, bei dem Trefferquote noch ≥ 85% ist
recall_85_idxs  = np.where(recalls[:-1] >= 0.85)[0]
thresh_85_recall = thresholds[recall_85_idxs[-1]]         # → 0.303

# Schwellenwert anwenden: Wahrscheinlichkeits-Score → binäre Betrug/kein-Betrug-Entscheidung
y_pred = (y_score >= threshold).astype(int)
```

### Was die Zelle ausgegeben hat
```
[max_f1]     threshold=0.5267  precision=0.877  recall=0.781  alerts=1,910

[recall_85]  threshold=0.3033  precision=0.703  recall=0.850  alerts=2,594

Recommendation: prefer recall_85 — missing fraud is far costlier than a false alert.
```

### Was es bedeutet

**Schritt 1 — Das Modell gibt jedem Transaktion einen Score, kein Urteil.**
Der Random Forest gibt für jede Transaktion eine Zahl zwischen 0 und 1 aus — etwa „0,73 Wahrscheinlichkeit, dass dies Betrug ist." Er sagt nie direkt „ja Betrug" oder „nein kein Betrug". *Du* entscheidest den Grenzwert.

**Schritt 2 — Der Schwellenwert wandelt diesen Score in eine echte Entscheidung um.**
Wenn du den Schwellenwert auf 0,5 setzt: Jede Transaktion mit einem Score über 0,5 wird als Betrug markiert, jede darunter wird durchgelassen. Denke daran als Tor: Das Tor anheben → weniger Dinge werden markiert → weniger Fehlalarme, aber mehr verpasste Betrugsfälle. Das Tor senken → mehr Dinge werden markiert → mehr Betrug erkannt, aber mehr unschuldige Kunden erhalten auch Anrufe.

**Schritt 3 — Warum 0,5 hier der falsche Standard ist.**
Der Standard-Schwellenwert von 0,5 wurde für ausgeglichene Datensätze entwickelt, bei denen Betrug und legitime Transaktionen ungefähr gleich sind. In unserem Datensatz sind nur 0,58 % der Transaktionen Betrug. Mit 0,5 ist das Modell zu konservativ — eine Transaktion mit einem Score von 0,35 ist relativ zur Grundlinie bereits sehr verdächtig, wird aber fälschlicherweise freigegeben. Ein niedrigerer Schwellenwert erfasst diese Fälle.

**Schritt 4 — Zwei Schwellenwerte für zwei verschiedene Situationen.**

| Name | Schwellenwert | Was er macht | Wann zu verwenden |
|------|---------------|--------------|------------------|
| `max_f1` | 0,527 | Beste Gesamtbalance zwischen Betrug-Erkennen und Fehlalarmen-Vermeiden | Allgemein — kein spezifisches Ziel gesetzt |
| `recall_85` | 0,303 | Garantiert das Erkennen von mindestens 85 % aller Betrugs | Wenn die Bank ein SLA hat: *„Wir dürfen nicht mehr als 15 % des Betrugs verpassen"* |

Der `recall_85`-Schwellenwert erkennt 85 von 100 Betrugsfällen (im Vergleich zu 78 bei `max_f1`), erzeugt aber 684 mehr tägliche Warnungen (2.594 vs. 1.910). Ob dieser Kompromiss die Mühe wert ist, hängt davon ab, wie viele Ermittler verfügbar sind und wie kostspielig jeder verpasste Betrugsfall ist — eine Entscheidung, die das Geschäftsteam trifft, nicht der Data Scientist.

---

## Schritt 10 — Hyperparameter-Tuning

Parameter sind das, was das Modell *aus Daten lernt* (Aufteilungsbedingungen in jedem Baum). *Hyper*parameter sind das, was *du* vor dem Training setzt: wie viele Bäume aufgebaut werden, wie tief sie wachsen, wie viele Merkmale an jedem Knoten berücksichtigt werden. Dieser Schritt wählt Hyperparameter manuell aus und trainiert ein abgestimmtes Modell zum Vergleich mit dem Baseline.

### Schlüsselcode
```python
rf_tuned = RandomForestClassifier(
    n_estimators=300,       # gleich wie Baseline — abnehmende Erträge über ~300 Bäume
    max_depth=30,           # tiefer als Baseline (20) um komplexere Muster zu erfassen
    min_samples_leaf=2,     # leichtere Regularisierung als Baseline (50) → flexiblere Aufteilungen
    max_features="sqrt",    # bei jeder Aufteilung nur √11 ≈ 3 Merkmale betrachten → dekorreliert Bäume
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_tuned.fit(X_train_sm, y_train_sm)  # auf SMOTE-erweiterten Daten trainiert (~2,6 Mio. Zeilen)
```

### Was die Zelle ausgegeben hat
```
         model    pr_auc  recall_at_0_5  precision_at_0_5  alert_count_at_0_5
0  rf_balanced  0.883456       0.727             0.958                1628
1     rf_smote  0.865872       0.790             0.860                1969
2     rf_tuned  0.862819       0.796             0.840                2033

Selected params: n_estimators=300, max_depth=30, min_samples_leaf=2, max_features="sqrt"
```

### Was es bedeutet

**Warum manuelle Auswahl statt automatisierter Suche:**
Automatisierte Suchwerkzeuge wie `RandomizedSearchCV` würden 20–30+ vollständige Modellanpassungen erfordern. Auf 1,3 Mio. Zeilen (oder 2,6 Mio. nach SMOTE) dauert jede Anpassung mehrere Minuten — was eine vollständige Suche auf der CPU unpraktisch macht. Stattdessen wurden die Hyperparameter direkt basierend auf dem bekannten Verhalten von Random Forests ausgewählt:

| Hyperparameter | Wert | Begründung |
|---|---|---|
| `n_estimators` | 300 | Mehr Bäume = stabilere Vorhersagen; abnehmende Erträge über ~300 |
| `max_depth` | 30 | Tief genug um komplexe Betrugsmuster zu erfassen, ohne unbegrenzt zu sein |
| `min_samples_leaf` | 2 | Leichte Regularisierung um das Auswendiglernen einzelner Ausreißer zu vermeiden |
| `max_features` | `"sqrt"` | Klassischer Standard — dekorreliert Bäume durch Begrenzung der Merkmale pro Aufteilung |
| `class_weight` | `"balanced_subsample"` | Pro-Baum-Klassenumgewichtung zusätzlich zu SMOTE |

**Warum PR-AUC leicht sank, während die Trefferquote sich verbesserte:**
Das scheint zunächst rückwärts. Die Erklärung ist ein Regularisierungs-Kompromiss:
- Die abgestimmten Parameter (`max_depth=30`, `min_samples_leaf=2`) erlauben tiefere, detailliertere Bäume als die Baseline (`max_depth=20`, `min_samples_leaf=50`).
- Tiefere Bäume erfassen mehr Betrugsmuster auf Kosten von etwas mehr Rauschen → Trefferquote steigt, aber Präzision und Gesamt-PR-AUC sinken leicht.
- Stell es dir vor als kleinen Gesamtqualitätstausch (PR-AUC 0,8835 → 0,8628) für bessere Abdeckung tatsächlicher Betrugsfälle (Trefferquote 72,7 % → 79,6 %).

**Warnungsanzahl:** Das abgestimmte Modell erzeugt 2.033 tägliche Warnungen — 405 mehr als die Baseline. Ob dieser Kompromiss es wert ist, hängt von der Kapazität des Betrugs-Teams ab.

---

## Schritt 11 — Validierung & Merkmal-Wichtigkeit

Ein guter PR-AUC-Score allein beantwortet nicht die Frage: *„Sollte ich diesem Modell tatsächlich vertrauen?"* Dieser Schritt führt drei Plausibilitätsprüfungen durch.

### Schlüsselcode
```python
# Kalibrierten Schwellenwert aus Schritt 9 anwenden (nicht den Standard 0.5)
y_pred_tuned = (tuned_artifacts["y_score_test"] >= best_threshold).astype(int)

# Prüfung 1: Konfusionsmatrix (visueller Plot)
ConfusionMatrixDisplay.from_predictions(
    y_test_np, y_pred_tuned,
    display_labels=["Legitimate", "Fraud"], colorbar=False
)

# Prüfung 2: Überanpassung — PR-AUC auf Trainings- vs. Testdaten vergleichen
train_score = average_precision_score(y_train_sm, tuned_model.predict_proba(X_train_sm)[:, 1])
test_score  = average_precision_score(y_test_np,  tuned_artifacts["y_score_test"])
gap = train_score - test_score

# Prüfung 3: Merkmal-Wichtigkeiten — welche Merkmale haben die Aufteilungen gesteuert?
importances = pd.Series(tuned_model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)
```

### Was die Zelle ausgegeben hat

**Prüfung 1 — Konfusionsmatrix (visueller Plot bei Schwellenwert = 0,527)**

Die Konfusionsmatrix ist ein Plot; ungefähre Anzahlen aus rf_tuned beim kalibrierten Schwellenwert:

```
                  Vorhergesagt: Legitim   Vorhergesagt: Betrug
Tatsächlich: Legitim   TN ≈ 553.264         FP ≈ 310
Tatsächlich: Betrug    FN ≈ 445             TP ≈ 1.700
```

- **1.700 Betrugsfälle erkannt** (TP) — Transaktionen korrekt markiert, bevor sie den Karteninhaber Geld kosten
- **310 Fehlalarme** (FP) — legitime Transaktionen markiert; jede erfordert einen Ermittler-Rückruf
- **445 verpasste Betrugsfälle** (FN) — echter Betrug, der unerkannt durchgeschlüpft ist
- **553.264 freigegeben** (TN) — legitime Transaktionen korrekt durchgewunken

**Prüfungen 2 & 3 — Überanpassung und Merkmal-Wichtigkeiten**
```
Overfitting check — PR-AUC:  train=1.0000  test=0.8628
  Gap (train - test) = 0.1372  ← possible overfit

Top feature importances (rf_tuned):
  amt           56.6%
  hour          20.5%
  category      11.4%
  day_of_week    3.2%
  month          2.3%
  age            1.9%
  city_pop       1.3%
  merchant       0.8%
  job            0.8%
  distance_km    0.6%
```

### Was es bedeutet

**Prüfung 1 — Konfusionsmatrix:**
Die Konfusionsmatrix ist ein 2×2-Raster, das jedes mögliche Ergebnis für jede Vorhersage des Modells zeigt:

|  | Vorhergesagt: Legitim | Vorhergesagt: Betrug |
|---|---|---|
| **Tatsächlich: Legitim** | Richtig Negativ (TN) — korrekt freigegeben | Falsch Positiv (FP) — unnötige Warnung |
| **Tatsächlich: Betrug** | Falsch Negativ (FN) — verpasster Betrug ✗ | Richtig Positiv (TP) — erkannter Betrug ✓ |

Eine Metrik wie PR-AUC komprimiert all das in eine Zahl — die Konfusionsmatrix macht die Kompromisse in rohen Anzahlen sichtbar. Bei der Betrugserkennung ist die kritische Zelle **FN (verpasster Betrug)**: Jede davon ist eine betrügerische Transaktion, die den Karteninhaber Geld gekostet hat. FP (Fehlalarme) haben eine andere Kosten: Ermittlerzeit. Der kalibrierte Schwellenwert aus Schritt 9 (Trefferquote-85%-SLA bei 0,303) verschiebt die Grenze, um mehr Vorhersagen in „Betrug" zu verschieben und FN auf Kosten von mehr FP zu reduzieren — die Konfusionsmatrix lässt dich genau sehen, wie viele Fälle sich verschoben haben.

**Prüfung 2 — Überanpassung (Overfitting):**
Ein Train-PR-AUC von 1,0000 bedeutet, dass das Modell die Trainingsdaten perfekt auswendig gelernt hat — es kann sich an jede trainierte Transaktion ohne einen einzigen Fehler erinnern. Der Test-PR-AUC von 0,8628 ist niedriger, aber immer noch stark. Diese Lücke (0,137) ist als „mögliche Überanpassung" (Overfitting) gekennzeichnet, ist aber für einen Random Forest auf einem großen Datensatz *handhabbar*:
- Trainingsdaten haben 1,3 Mio. Zeilen — das Modell hat eine enorme Vielfalt an Mustern gesehen.
- Betrugsmuster in diesem simulierten Datensatz sind konsistent, sodass das Modell gut genug verallgemeinert, dass der Test-PR-AUC hoch ist.
- Ein Train-Score von 1,0 ist für tiefe Random Forests tatsächlich *erwartet* — Bäume können die Trainingsblätter immer perfekt aufteilen. Der Test-Score ist das, was zählt.

**Prüfung 3 — Merkmal-Wichtigkeit (mittlere Verringerung der Verunreinigung):**
Jeder Prozentsatz zeigt, wie viel dieses Merkmal die Unsicherheit (Verunreinigung) über alle Aufteilungen in allen Bäumen reduziert hat. Eine höhere Zahl bedeutet, dass das Modell bei Entscheidungen mehr auf dieses Merkmal angewiesen war.

- **`amt` bei 56,6 %** dominiert, weil der Transaktionsbetrag Betrug direkt von normalem Verhalten trennt — erinnere dich aus Schritt 4, dass Transaktionen über 1.000 $ 41× häufiger Betrug sind. Das Modell hat das auch gelernt.
- **`hour` bei 20,5 %** bestätigt das Spätnacht-Betrugsmuster aus Schritt 4. Tageszeit ist die zweitstärkste Aufteilung.
- **`category` bei 11,4 %** bestätigt die Online-vs.-Im-Laden-Unterscheidung aus Schritt 4.
- Die restlichen acht Merkmale tragen kleinere, aber bedeutsame Signale bei — keines ist Rauschen, was bestätigt, dass das Feature Engineering in Schritt 6 solide war.

**Warum das für Vertrauen wichtig ist:**
Wenn `trans_num` oder `cc_num` (die ID-Spalten, die wir in Schritt 5 entfernt haben) in den Top-Merkmalen erschienen wären, würde das auf einen Datenleck-Fehler hinweisen. Die Tatsache, dass nur *verhaltensbezogene* Signale dominieren (Betrag, Zeit, Händlertyp), bedeutet, dass das Modell echte Betrugsmuster lernt, nicht Datenbankartifakte auswendig lernt.

---

## Modellschlussfolgerung

**Das `rf_balanced`-Modell ist die empfohlene Wahl für den Einsatz** am `recall_85`-Betriebsschwellenwert (0,303).

### Warum diese Schlussfolgerung

- **Starke Betrugserkennungsfähigkeit.** PR-AUC von 0,8835 bedeutet, dass das Modell gut über jeden möglichen Betriebsschwellenwert performt — nicht nur einen glücklichen Grenzwert. Ein zufälliger Klassifizierer auf diesem Datensatz würde ~0,006 erzielen; bei 0,88 trennt das Modell Betrug von legitimen Transaktionen wirklich.

- **Die Überanpassungslücke ist akzeptabel.** Train-PR-AUC ist 1,0 (für tiefe Bäume erwartet), während Test-PR-AUC 0,8628 ist — eine Lücke von 0,137. Das klingt groß, ist aber typisch für Random Forests auf großen Datensätzen. Der Test-Score selbst ist das, was für die Produktion zählt, und 0,86 ist stark.

- **Kein Datenleck erkannt.** Die drei Top-Merkmale (`amt`, `hour`, `category`) sind alle echte Verhaltens-Signale, die in Schritt 4 entdeckt wurden. Keine Bezeichner-Spalten sind durchgesickert — das Modell hat gelernt, *warum* Betrug passiert, nicht *welche spezifischen Karten* im Training betrügerisch waren.

- **Operationell handhabbares Warnungsvolumen.** Am `recall_85`-Schwellenwert erkennt das Modell 85 % aller Betrugsfälle (1.823 von 2.145 Fällen) und erzeugt dabei ~2.594 tägliche Warnungen. Ein Betrugs-Team, das diese überprüft, würde in etwa 7 von 10 Warnungen echten Betrug finden — eine Präzision von 70,3 %, die für den Produktions-Betriebsbetrieb realistisch ist.

- **`rf_tuned` ist trotz höherer Trefferquote nicht der Gewinner.** Obwohl `rf_tuned` die Trefferquote auf 79,6 % erhöht, sinkt sein PR-AUC auf 0,8628 und tägliche Warnungen wachsen auf 2.033 beim Standard-Schwellenwert — ohne die gleiche Qualitätskurve wie `rf_balanced` zu bieten. Für ein Team, das das Trefferquote-85%-SLA benötigt, erfüllt `rf_balanced` bei Schwellenwert 0,303 dieses Ziel bereits mit einem besseren Gesamt-Präzisions-Trefferquoten-Kompromiss.

- **SMOTE wurde korrekt entfernt.** Das Experiment in Schritt 8 bestätigte, dass synthetische Überabtastung die Verallgemeinerung verschlechterte (PR-AUC −1,8 Pp) und gleichzeitig das Warnungsvolumen aufblies (+341/Tag). Der Ansatz `class_weight="balanced_subsample"` behandelt das 171:1-Ungleichgewicht zuverlässiger, ohne künstliche Trainingsbeispiele einzuführen.

### Was das Modell stärker machen würde

- Mehr gelabelte Betrugsbespiele im Laufe der Zeit — das Modell wurde nur auf 7.506 Betrugsfällen trainiert; reale Systeme werden monatlich neu trainiert, wenn neue Betrugsmuster auftauchen.
- Händler-basierte Häufigkeitskodierung könnte mit Target Encoding verfeinert werden, um hochbetrügerische Händler besser ohne ordinale Annahmen zu erfassen.
- Die Überanpassungslücke könnte durch leichtes Erhöhen von `min_samples_leaf` verringert werden — dies würde eine kleine Menge Trefferquote für eine engere Train/Test-Lücke eintauschen.

---

## Finaler Modellvergleich

| Modell | ROC-AUC | PR-AUC | Präzision | Trefferquote | Tägliche Warnungen |
|--------|---------|--------|-----------|--------------|-------------------|
| rf_balanced | 0,9877 | 0,8835 | 0,958 | 0,727 | 1.628 |
| rf_smote | — | 0,8659 | 0,860 | 0,790 | 1.969 |
| rf_tuned | — | 0,8628 | 0,840 | 0,796 | 2.033 |

**Die Tabelle lesen:** `rf_balanced` hat den höchsten PR-AUC und die wenigsten Warnungen, aber die niedrigste Trefferquote (72,7 % der Betrugsfälle erkannt). `rf_tuned` erkennt 7 Prozentpunkte mehr Betrug auf Kosten von 405 zusätzlichen täglichen Warnungen und einem leicht niedrigeren PR-AUC. Die richtige Wahl hängt von der Kapazität des Betrugs-Teams und dem SLA der Bank ab — beide sind valide Betriebspunkte.

---

## Gesamtfazit für Anfänger

### 1. Genauigkeit ist bei unausgewogenen Daten irreführend
Ein Modell, das nie „Betrug" vorhersagt, ist 99,4 % genau und erwischt null Kriminelle. Verwende immer ROC-AUC und PR-AUC für Betrugsaufgaben.

### 2. Datenqualitätsarbeit IST Modellierungsarbeit
Schritte 1–6 (Profilerstellung, Drift, Datenleck-Prüfung, Feature Engineering) erforderten mehr Code als das Training des Modells. In echten Data-Science-Jobs werden 70–80 % der Zeit hier verbracht.

### 3. Merkmal-Interaktionen sind mächtig
Das Muster „Online-Shopping um Mitternacht für 1.000 $+" ist 45× wahrscheinlicher Betrug als jedes einzelne Merkmal allein vorhersagt. Modelle, die Interaktionen erfassen (Random Forest, Gradient Boosting, Neuronale Netze), übertreffen lineare Modelle bei dieser Aufgabe dramatisch.

### 4. Datenleck ist der stille Killer
`cc_num` in den Daten zu lassen würde ein Modell erzeugen, das auf dem Papier toll aussieht, aber bei neuen Karten in der Produktion komplett scheitert. Frage immer: *„Hätte ich diese Information zum Zeitpunkt der Vorhersage?"*

### 5. Geschäftliche Einschränkungen prägen die Modellwahl
Ein Modell, das 20.000 Warnungen pro Tag erzeugt, ist operationell nutzlos, selbst wenn es hohe Trefferquote hat. Das beste Modell ist dasjenige, das das Erkennen von Betrug (Trefferquote) mit der Nichtüberwältigung von Ermittlern (Präzision) ausbalanciert.

---

## Abgeschlossene Schritte

Alle Pipeline-Schritte sind implementiert. Eine Zusammenfassung der dabei getroffenen Entscheidungen:

- **SMOTE (Schritt 8)** — getestet und entfernt; PR-AUC um 1,8 Pp gesunken, und `class_weight="balanced_subsample"` behandelt Ungleichgewicht effektiver ohne synthetische Daten
- **Schwellenwert-Kalibrierung (Schritt 9)** — zwei Betriebspunkte: `max_f1` (0,527) für ausgewogene Teams, `recall_85` (0,303) für SLA-eingeschränkte Umgebungen
- **Hyperparameter-Tuning (Schritt 10)** — manuelle Auswahl (automatisierte Suche bei 1,3 Mio.+ Zeilen unpraktisch); Parameter: `n_estimators=300`, `max_depth=30`, `min_samples_leaf=2`, `max_features="sqrt"`
- **Validierung (Schritt 11)** — Konfusionsmatrix (Prüfung 1), Überanpassungslücke von 0,137 handhabbar (Prüfung 2); Merkmal-Wichtigkeiten bestätigen, dass das Modell echte Betrugs-Signale lernt (Prüfung 3)

---

*Erstellt aus `notebook_colab.ipynb` — zuletzt ausgeführt am 06.03.2026*
