"""Generate a SYNTHETIC, news-like dataset for teaching. This is not real news.

Added on 2026-10-13. The original teaching dataset (fake_news_dataset.csv, described in
dataset_description.md) is not in this repository, so this script rebuilds a dataset with the
same 7 columns the notebook expects. It does not recreate the original rows.

Everything here is invented:
  * outlets and author names are fictional (any resemblance to real ones is accidental);
  * events, places, numbers and quotes are random combinations of template sentences;
  * "fake" articles differ from "real" ones only by writing style (sensational wording,
    capital letters, exclamation marks, vague sources), and the two styles overlap on purpose:
    about 15% of articles of each label are written in the other label's style.

Results measured on this data say nothing about real-world fake-news detection.

Usage:
    python data/generate_synthetic.py                      # 20,000 rows, seed 42
    python data/generate_synthetic.py --n 2000 --seed 7 --out /tmp/small.csv

The output is deterministic for a given seed, row count and overlap (only the Python standard
library is used). The script prints the SHA-256 of the file it wrote.
"""
import argparse
import csv
import datetime
import hashlib
import random
from pathlib import Path

SEED = 42
N_ROWS = 20_000
ANCHOR_DATE = datetime.date(2026, 10, 1)  # fixed, so the "past 3 years" window never drifts
MISSING_RATE = 0.05

CATEGORIES = {
    "Politics": dict(
        subjects=["the city council", "the transport minister", "the regional assembly", "the budget committee",
                  "the election commission", "a coalition of mayors", "the housing ministry", "the senate panel"],
        topics=["a new housing bill", "the annual budget", "a voting reform plan", "a public transport levy",
                "a proposed tax change", "a school funding formula", "a border management agreement",
                "a local election timetable"]),
    "Health": dict(
        subjects=["the health department", "a team of researchers", "the national hospital board",
                  "a university clinic", "the vaccine advisory group", "a regional health trust",
                  "the nursing association", "a public health agency"],
        topics=["a seasonal flu programme", "a diabetes screening trial", "a new sleep study",
                "hospital waiting times", "a childhood nutrition survey", "a dental care initiative",
                "an allergy treatment review", "a walking-for-health campaign"]),
    "Technology": dict(
        subjects=["a software company", "the standards body", "a team of engineers", "a robotics start-up",
                  "the telecom regulator", "an open-source foundation", "a chip manufacturer",
                  "a cybersecurity firm"],
        topics=["a battery design", "a wireless standard", "a language-learning app", "a data-centre cooling system",
                "a smart-meter rollout", "a password manager update", "a satellite internet service",
                "a voice assistant feature"]),
    "Business": dict(
        subjects=["the retail group", "a regional bank", "the airline", "a logistics firm", "the trade body",
                  "a furniture maker", "the energy supplier", "a food distributor"],
        topics=["quarterly earnings", "a new warehouse", "a merger proposal", "changes to delivery pricing",
                "a supplier contract", "a hiring plan", "an exchange-rate hedge", "a product recall notice"]),
    "Science": dict(
        subjects=["astronomers", "a marine biology lab", "the geology institute", "a physics collaboration",
                  "an international research group", "a climate modelling centre", "a botany department",
                  "a materials science team"],
        topics=["a distant star cluster", "deep-sea sediment samples", "a new alloy", "bird migration patterns",
                "an ice-core record", "a particle detector upgrade", "soil bacteria", "a comet observation"]),
    "Sports": dict(
        subjects=["the home side", "the national federation", "a veteran coach", "the league office",
                  "a young striker", "the athletics club", "the tournament organisers", "the rowing team"],
        topics=["a late equaliser", "the transfer window", "a stadium renovation", "the season fixtures",
                "a training camp", "a ticketing change", "a youth academy", "an injury update"]),
    "Entertainment": dict(
        subjects=["the film festival", "a touring orchestra", "the studio", "a debut novelist",
                  "the theatre company", "a documentary crew", "the museum", "a streaming service"],
        topics=["a new season announcement", "an exhibition opening", "a soundtrack release", "a casting decision",
                "a retrospective screening", "a translated novel", "a restored silent film", "a summer tour"]),
    "Environment": dict(
        subjects=["the water authority", "a conservation charity", "the forestry service", "a coastal council",
                  "the weather office", "a farmers' cooperative", "the parks department", "a recycling consortium"],
        topics=["a river clean-up", "a tree-planting scheme", "a flood barrier", "a drought warning",
                "a wetland survey", "a plastics collection pilot", "a solar farm plan", "a beach erosion report"]),
}

PLACES = ["Eastvale", "Marlow Bay", "Kingsford", "Ashgrove", "Northhaven", "Redcliff", "Port Lowen", "Stonebridge",
          "Fairmont", "Westerly", "Brindle Heath", "Haverton", "Silverlake", "Coldwater", "Merrow", "Quillham"]
REAL_OUTLETS = ["Harbor Ledger", "Meridian Daily", "Northfield Post", "Granite Courier", "Brightwater Times",
                "Oakridge Herald", "Lantern Wire", "Cobalt Chronicle"]
FAKE_OUTLETS = ["TruthBeacon Daily", "The Real Wire 24", "WakeUp Gazette", "Patriot Pulse Online",
                "Daily Revelation", "Unfiltered Feed"]
FIRST = ["Alex", "Maria", "Samir", "Chloe", "Daniel", "Priya", "Tomas", "Hannah", "Kofi", "Elena", "Marcus", "Aiko",
         "Jonas", "Leila", "Owen", "Sofia"]
LAST = ["Hartley", "Novak", "Okafor", "Lindqvist", "Moreau", "Banerjee", "Castillo", "Whitfield", "Duarte",
        "Petrov", "Nakamura", "Fletcher", "Haddad", "Kowalski", "Brennan", "Osei"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Neutral reporting style (mostly "real")
CALM_SENTENCES = [
    "{subject} said on {weekday} that {topic} would be reviewed over the coming weeks.",
    "In a statement released in {place}, {subject} described {topic} as a routine step.",
    "According to figures published in {month}, about {pct} percent of people surveyed supported {topic}.",
    "A spokesperson for {subject} told reporters that no final decision on {topic} has been taken.",
    "The announcement follows a consultation that began in {month} and drew {num} written responses.",
    "Officials in {place} noted that earlier plans for {topic} had been delayed by about {weeks} weeks.",
    "Analysts quoted in the report said the effect of {topic} would likely be modest in the first year.",
    "The document, which runs to {num} pages, sets out the timetable and cost estimates for {topic}.",
    "Critics said more detail was needed, while supporters argued that {topic} was overdue.",
    "{subject} said it would publish an update on {topic} once the figures have been checked.",
    "Local residents in {place} were invited to comment on {topic} at a public meeting on {weekday}.",
    "The latest data, compiled from {num} records, show a change of {pct} percent compared with last year.",
    "A review of {topic} is expected to report back by the end of {month}, officials said.",
    "{subject} added that the cost of {topic} would be met from existing budgets.",
    "Earlier this year, {subject} had said that {topic} was under discussion but not yet agreed.",
    "The figures have not been independently verified, the report noted, and may be revised.",
]
# Sensational style (mostly "fake")
LOUD_SENTENCES = [
    "SHOCKING: {subject} has been hiding the TRUTH about {topic} and nobody is talking about it!",
    "Sources close to {subject} (who cannot be named) reveal that {topic} is a total cover-up.",
    "You will NOT believe what really happened in {place} after {topic} was announced!!!",
    "Experts they do not want you to hear from say {topic} will change EVERYTHING overnight.",
    "Share this before it gets deleted: {subject} has finally been caught lying about {topic}.",
    "Everyone in {place} already knows the real story behind {topic}, but the mainstream media stays silent.",
    "A leaked memo PROVES that {topic} was planned months ago, and the numbers they gave you were fake.",
    "This is the biggest scandal of the decade and {subject} is panicking over {topic}.",
    "Insiders claim that {pct} percent of the public has been deceived about {topic}, and it is getting worse!",
    "Wake up! {topic} is being used to control you, and {subject} will never admit it.",
    "It is a total disaster, an absolute outrage, and anyone who disagrees about {topic} is part of the problem.",
    "Do your own research on {topic} because the so-called experts are paid to keep quiet.",
    "BREAKING: {subject} makes a stunning U-turn on {topic} and the internet is exploding!",
    "No one can explain why {subject} refuses to answer simple questions about {topic}.",
    "A brave whistleblower in {place} says the truth about {topic} will leave you speechless.",
    "They tried to silence this story about {topic}, but it is spreading everywhere right now!",
]
# Shared connective filler, used by both styles
NEUTRAL_FILLER = [
    "The matter has attracted attention across {place} in recent days.",
    "More information is expected later this week.",
    "The story is developing and further details may follow.",
    "Several people in {place} said they had followed the discussion closely.",
    "The topic has come up at meetings in {place} several times this year.",
]

CALM_TITLES = ["{subject_short} reviews {topic_short}", "{topic_short} under review in {place}",
               "What {subject_short} said about {topic_short}", "{place}: {topic_short} timetable set out",
               "{subject_short} publishes update on {topic_short}"]
LOUD_TITLES = ["SHOCKING truth about {topic_short} revealed", "{subject_short} caught hiding {topic_short}!",
               "You will not believe {topic_short} in {place}", "The cover-up behind {topic_short} exposed",
               "BREAKING: {topic_short} bombshell in {place}"]


def _short(phrase):
    for article in ("the ", "a ", "an "):
        if phrase.startswith(article):
            return phrase[len(article):]
    return phrase


def _fill(rng, template, subject, topic, place):
    return template.format(
        subject=subject[0].upper() + subject[1:] if template.startswith("{subject}") else subject,
        topic=topic, place=place, weekday=rng.choice(WEEKDAYS), month=rng.choice(MONTHS),
        pct=rng.randint(3, 78), num=rng.randint(12, 950), weeks=rng.randint(2, 14),
        subject_short=_short(subject).capitalize(), topic_short=_short(topic))


def make_article(rng, label, overlap):
    """Return (title, text). With probability `overlap` the article is written in the other style."""
    category = rng.choice(list(CATEGORIES))
    pool = CATEGORIES[category]
    subject, topic, place = rng.choice(pool["subjects"]), rng.choice(pool["topics"]), rng.choice(PLACES)
    loud = (label == "fake") != (rng.random() < overlap)
    # Loud articles mix in some calm sentences and vice versa, so single sentences never decide the label
    p_loud_sentence = 0.55 if loud else 0.08
    target_words = rng.randint(200, 300)
    sentences, words = [], 0
    while words < target_words:
        r = rng.random()
        if r < 0.12:
            template = rng.choice(NEUTRAL_FILLER)
        elif rng.random() < p_loud_sentence:
            template = rng.choice(LOUD_SENTENCES)
        else:
            template = rng.choice(CALM_SENTENCES)
        sentence = _fill(rng, template, subject, topic, place)
        sentences.append(sentence)
        words += len(sentence.split())
    title_template = rng.choice(LOUD_TITLES if loud else CALM_TITLES)
    title = _fill(rng, title_template, subject, topic, place)
    title = title[0].upper() + title[1:]
    return category, title, " ".join(sentences)


def generate(n=N_ROWS, seed=SEED, overlap=0.15):
    rng = random.Random(seed)
    labels = ["real"] * (n // 2) + ["fake"] * (n - n // 2)
    rng.shuffle(labels)
    rows = []
    for label in labels:
        category, title, text = make_article(rng, label, overlap)
        outlets = FAKE_OUTLETS if (label == "fake") != (rng.random() < 0.2) else REAL_OUTLETS
        source = rng.choice(outlets)
        author = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        date = ANCHOR_DATE - datetime.timedelta(days=rng.randint(0, 3 * 365))
        if rng.random() < MISSING_RATE:
            source = ""
        if rng.random() < MISSING_RATE:
            author = ""
        rows.append({"title": title, "text": text, "date": date.isoformat(), "source": source,
                     "author": author, "category": category, "label": label})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=N_ROWS, help="number of rows (default 20000)")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (default 42)")
    parser.add_argument("--overlap", type=float, default=0.15,
                        help="share of articles written in the other label's style (default 0.15)")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "fake_news_dataset.csv"))
    args = parser.parse_args()
    rows = generate(args.n, args.seed, args.overlap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text", "date", "source", "author", "category", "label"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"SYNTHETIC teaching data, seed={args.seed}, overlap={args.overlap}")
    print(f"wrote {out} ({len(rows)} rows, {out.stat().st_size:,} bytes)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
