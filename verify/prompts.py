# SPDX-License-Identifier: CC0-1.0
"""Fixed prompts for verification. Raw text (no chat template) so the comparison
does not depend on template handling. One prompt is longer than the 513-token
sliding window on purpose, to exercise the window boundary and cache rotation."""

PROMPTS = {
    "de_short": "Die Hauptstadt der Schweiz ist Bern. Die grösste Stadt des Landes ist",
    "en_short": "The quick brown fox jumps over the lazy dog. The capital of France is",
    "de_code": (
        "Schreibe eine Python-Funktion, die prüft, ob eine Zahl eine Primzahl ist.\n\n"
        "def ist_primzahl(n):\n    "
    ),
    "en_long": (
        "The history of the bicycle begins in the early nineteenth century, when a German "
        "inventor named Karl Drais built a two-wheeled, steerable, human-propelled machine that "
        "he called the Laufmaschine, or running machine. It had no pedals; the rider sat on a "
        "wooden frame and pushed along the ground with their feet, coasting whenever the road "
        "sloped downhill. The machine was patented in 1818 and enjoyed a brief fashion in Europe "
        "before being banned from many pavements because riders kept colliding with pedestrians. "
        "Roughly forty years later, French mechanics added cranks and pedals to the front wheel, "
        "producing what came to be known as the velocipede or, more colloquially, the boneshaker, "
        "on account of its iron tyres and the appalling roads of the period. Because the pedals "
        "were fixed directly to the front axle, the only way to go faster was to make the front "
        "wheel larger, and so the high-wheeler, or penny-farthing, emerged in the 1870s with a "
        "driving wheel sometimes exceeding one and a half metres in diameter. These machines were "
        "fast but dangerous: the rider sat almost directly above the axle and a sudden stop would "
        "pitch them head first over the handlebars, an accident so common that it acquired its own "
        "name, the header. The decisive breakthrough came in 1885 with the Rover safety bicycle "
        "designed by John Kemp Starley in Coventry. It had two wheels of similar size, a diamond "
        "frame, and a chain drive to the rear wheel, which allowed the gear ratio to be chosen "
        "independently of wheel size. Three years later John Boyd Dunlop reinvented the pneumatic "
        "tyre, and the combination of the safety frame and air-filled tyres produced a machine "
        "recognisably the same as the one ridden today. The 1890s saw a bicycle boom on both sides "
        "of the Atlantic. Manufacturing techniques developed for bicycles, including ball bearings, "
        "tension-spoked wheels, steel tubing and chain drives, were later taken up by the makers of "
        "motorcycles and automobiles, and several early aircraft builders, among them the Wright "
        "brothers, began as bicycle mechanics. The bicycle also had a notable social effect. It gave "
        "working people affordable personal mobility for the first time, widened the radius within "
        "which young people met and married, and became closely associated with the movement for "
        "women's emancipation, since it required practical clothing and offered independence from "
        "chaperones. Over the twentieth century the basic design changed little, although derailleur "
        "gears, aluminium and later carbon-fibre frames, and improved brakes made bicycles lighter "
        "and more versatile. In many cities the bicycle was displaced by the motor car after the "
        "Second World War, only to return from the 1970s onwards as concerns about congestion, "
        "pollution and public health grew. Today more bicycles are produced each year than cars, and "
        "in cities such as Amsterdam and Copenhagen the majority of daily trips are made by bike. "
        "The machine that began as a wooden toy for aristocrats has become one of the most widely "
        "used forms of transport in the world. In summary, the main stages of this development were"
    ),
    "de_long": (
        "Die Geschichte des Fahrrads beginnt im frühen neunzehnten Jahrhundert, als der badische "
        "Forstbeamte Karl Drais eine zweirädrige, lenkbare Laufmaschine baute, die er 1817 auf einer "
        "Fahrt von Mannheim nach Schwetzingen der Öffentlichkeit vorführte. Pedale gab es noch nicht; "
        "der Fahrer sass auf einem hölzernen Rahmen und stiess sich mit den Füssen vom Boden ab. Die "
        "Erfindung fiel in eine Zeit der Not, denn nach dem Ausbruch des Vulkans Tambora war das "
        "Jahr 1816 als Jahr ohne Sommer in die Geschichte eingegangen, die Ernten waren verdorben, "
        "und Pferde zu halten war teuer geworden. Die Laufmaschine war daher auch als Ersatz für das "
        "Reitpferd gedacht. Sie fand zunächst einige Nachahmer in Frankreich und England, verschwand "
        "aber bald wieder, weil sie auf den schlechten Strassen unbequem war und auf den Gehwegen "
        "verboten wurde. Erst in den 1860er Jahren brachten französische Mechaniker Tretkurbeln am "
        "Vorderrad an und schufen damit das Veloziped, das wegen seiner eisenbereiften Räder im "
        "Volksmund auch Knochenschüttler hiess. Da die Übersetzung direkt vom Durchmesser des "
        "Vorderrads abhing, wurden die Räder immer grösser, bis das Hochrad mit einem Vorderrad von "
        "über eineinhalb Metern entstand. Das Hochrad war schnell, aber gefährlich, denn bei einem "
        "plötzlichen Halt stürzte der Fahrer kopfüber nach vorn. Die entscheidende Wende brachte 1885 "
        "das Sicherheitsniederrad von John Kemp Starley aus Coventry mit zwei etwa gleich grossen "
        "Rädern, einem Diamantrahmen und einem Kettenantrieb auf das Hinterrad. Drei Jahre später "
        "erfand John Boyd Dunlop den Luftreifen neu, und aus der Verbindung beider Neuerungen "
        "entstand ein Fahrzeug, das sich vom heutigen Fahrrad kaum unterscheidet. In den 1890er Jahren "
        "erlebte das Fahrrad in Europa und Nordamerika einen regelrechten Boom. Viele Techniken, die "
        "für die Fahrradproduktion entwickelt wurden, etwa Kugellager, Speichenräder, Stahlrohre und "
        "Kettenantriebe, übernahmen später die Hersteller von Motorrädern und Automobilen; die Brüder "
        "Wright, die das erste Motorflugzeug bauten, hatten zuvor eine Fahrradwerkstatt betrieben. "
        "Auch gesellschaftlich hinterliess das Fahrrad Spuren. Es verschaffte Arbeitern erstmals eine "
        "bezahlbare persönliche Mobilität, vergrösserte den Umkreis, in dem junge Menschen einander "
        "kennenlernten, und wurde zu einem Symbol der Frauenbewegung, weil es praktische Kleidung "
        "verlangte und Unabhängigkeit von Begleitpersonen bot. Im zwanzigsten Jahrhundert änderte "
        "sich die Grundform kaum, doch Kettenschaltungen, Rahmen aus Aluminium und später aus "
        "Kohlefaser sowie bessere Bremsen machten die Räder leichter und vielseitiger. Nach dem "
        "Zweiten Weltkrieg wurde das Fahrrad in vielen Städten vom Auto verdrängt, kehrte aber ab den "
        "1970er Jahren zurück, als Staus, Luftverschmutzung und Fragen der Gesundheit an Bedeutung "
        "gewannen. Heute werden jedes Jahr mehr Fahrräder als Autos hergestellt, und in Städten wie "
        "Amsterdam oder Kopenhagen wird die Mehrheit der täglichen Wege mit dem Velo zurückgelegt. "
        "Zusammengefasst lassen sich die wichtigsten Etappen dieser Entwicklung wie folgt beschreiben:"
    ),
}


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Aleph-Alpha/Kolibri-1")
    for name, text in PROMPTS.items():
        print(name, len(tok.encode(text)))
