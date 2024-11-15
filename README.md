# **Prosjektrapport: Filmanbefalingssystem**

Applikasjonen kan finnes ved: [https://huggingface.co/spaces/Mariusbr/DAT158](https://huggingface.co/spaces/Mariusbr/DAT158)
Exempel kjøring av applikasjonen kan finnes ved filen Example.mp4
Prosjketet er lager av Marius Reikerås

## **Introduksjon**
Formålet med dette prosjektet er å utvikle en applikasjon for anbefaling av filmer basert på brukerens preferanser for sjangre og favorittfilmer. Applikasjonen er ment å gi brukerne tilpassede filmanbefalinger uten krav om innlogging eller deling av sensitive opplysninger. Dette er en enkel løsning som retter seg mot filmentusiaster som søker nye filmforslag basert på deres tidligere preferanser.

## **Business Objectives**
Prosjektet er rettet mot å løse et viktig problem for brukerne: det å finne interessante filmer å se på, basert på deres smak og tidligere erfaringer. Ved å bruke maskinlæring til å analysere brukernes preferanser, kan filmanbefalingssystemet:

1. **Tilby personlige anbefalinger**: Ved å bruke klustering og tekstanalyse kan systemet foreslå relevante filmer som brukerne sannsynligvis vil like.
2. **Forbedre brukeropplevelsen**: Tjenesten kan hjelpe brukere med å unngå tidkrevende søk etter interessante filmer ved å presentere tilpassede anbefalinger.
3. **Oppnå brukervennlighet uten behov for sensitiv informasjon**: Brukerne får tilgang til anbefalingene uten innlogging eller deling av personopplysninger, noe som øker tilgjengeligheten og opplevelsen for brukeren.

## **Business Impact**
Dette systemet har forretningsmessige påvirkninger:

- **Økt brukertilfredshet**: Med personlig tilpassede anbefalinger får brukerne forslag som passer deres interesser, noe som kan føre til en mer engasjerende filmopplevelse.
- **Forbedret beslutningsstøtte**: Ved å bruke dataanalyse for å gi tilpassede anbefalinger, kan systemet hjelpe brukere med å ta bedre valg når de skal velge en film.

## **Sammenligning med tilsvarende løsninger**
Tilsvarende løsninger, som større strømmeplattformer, krever ofte innlogging og bruker mer omfattende datainnsamling for å kunne tilby anbefalinger. Vår tjeneste gir brukerne rask tilgang til anbefalinger basert på enkle preferanseinput, noe som gjør løsningen mer tilgjengelig og mindre intrusiv.

Det finnes også utallige andre filmanbefalingsnettsteder, men dette prosjektet ble likevel valgt fordi ideen var interessant å utforske.

## **Business Metrics for Ytelsesmåling**
For å evaluere om systemet oppfyller sine forretningsmål, kan følgende business metrics brukes:

- **Brukertilfredshet**: Målt via tilbakemeldinger eller vurderinger fra brukerne.
- **Relevans**: Sikre at filmene som blir anbefalt er relevante for brukeren, ved å måle hvor ofte brukere finner anbefalingene interessante.
- **Systemtilgjengelighet**: Tjenesten bør ha en oppetid på minst 95 %, for å sikre pålitelig tilgang for brukerne.

## **Maskinlærings- og Software-Metrikker**
Flere tekniske metrikker er også relevante for å måle systemets ytelse:

- **Cosine Similarity Score**: Brukt for å finne hvor like filmer er basert på innholdet deres (tagger og sjangre). Dette målet brukes for å finne anbefalingene basert på brukerinput.
- **Antall vurderinger og gjennomsnittlig vurdering**: Disse faktorene blir brukt for å prioritere hvilke filmer som anbefales, men de blir ikke evaluert med spesifikke metrikker utover det å vektlegge popularitet og vurderingsgjennomsnitt.

## **Stakeholders**
Prosjektet involverer flere stakeholders:

- **Kunder/brukere**: Filmentusiaster som bruker tjenesten for å finne nye filmer å se på.
- **Utviklere**: Ansvarlige for å bygge og vedlikeholde maskinlæringsmodellen samt integrere den med brukergrensesnittet.

## **Tentativ Tidslinje med Milepæler**
Prosjektet følger en tentativ tidslinje med følgende milepæler:

1. **Prosjektoppstart og datainnsamling**: Definere mål og samle inn nødvendig filmdata.
2. **Modellutvikling**: Bygge og trene klusteringsmodellen for filmanbefalinger.
3. **Integrasjon med grensesnitt**: Bygge Gradio-grensesnittet for innsamling av brukerinput og generering av anbefalinger.
4. **Testing og rapportering**: Testing av modellens ytelse, dokumentering av resultater og levering av prosjektet.

## **Ressurser**
- **Personell**: Utvikler(e) ansvarlig for dataanalyse og maskinlæringsmodell.
- **Beregningsressurser**: Laptop for lokal utvikling og kjøring av modellen.
- **Hosting plattform**: Gradio ble brukt for å hoste applikasjonen slik at den ble tilgjengelig for alle.
- **Dataressurser**: MovieLens-datasett som inkluderer informasjon om filmer, vurderinger og brukertagger.

## **Data**
Dataene som brukes i dette prosjektet er hentet fra MovieLens 20M-datasettet, som inneholder vurderinger, filmer og tagger. Dataene er strukturerte og har variabler som film-ID, sjanger, brukervurderinger og tagger, som brukes til å lage anbefalingsmodellen. Prosjektet bruker følgende preprocessing-steg for å forberede dataene:

1. **Preprocessing av dataene**: Datasettet renses og preprocesses ved å fylle ut manglende verdier og kombinere relevante kolonner. For eksempel, sjangre og tagger kombineres til en samlet innholdsfunksjon.
2. **TF-IDF Vectorizer**: Tekstanalysemetoden TF-IDF brukes til å lage en representasjon av filmens innhold, basert på tagger og sjanger, som deretter brukes til klustering.
3. **K-Means-klustering**: Modellering skjer ved bruk av K-Means-klustering, som grupperer lignende filmer i klynger. Dette hjelper med å gi anbefalinger ved å finne filmer i samme klynge som de som brukeren liker.

**Modellering**
Det ble valgt K-means-klustering som hovedmetode for å gruppere filmer med lignende innholdskarakteristikker, som sjanger og brukergenererte tagger. Målet var å skape grupperinger som kunne brukes til å gi relevante anbefalinger basert på brukernes preferanser. Andre metoder ble også vurdert, men fant at klustering passet best for dette prosjektet.

**Evaluering av modellen**
- **Subjektiv vurdering**: Modellen ble evaluert ved manuelt å vurdere om anbefalingene var relevante og interessante for brukernes input. Selv om det ikke forelå en spesifikk sammenligningsmodell, ble resultatene vurdert ut fra om de møtte forventningene til relevans og variasjon i anbefalingene. 

- **Anbefalingsalgoritme**: Systemet kombinerer faktorer som likhet, popularitet (antall vurderinger) og gjennomsnittlig vurdering for å gi anbefalinger. Målet er å maksimere både relevans og nøyaktighet i anbefalingene. 

## **Deployment**
Filmanbefalingssystemet er implementert som et web-basert brukergrensesnitt ved hjelp av Gradio, som gir enkel interaksjon mellom brukeren og maskinlæringsmodellen. Brukeren kan skrive inn en liste med favorittfilmer og foretrukne sjangre, og systemet vil returnere anbefalte filmer. Modellen kjører lokalt, og Gradio grensesnittet sikrer en interaktiv og responsiv opplevelse for brukeren. Modellen distribueres uten behov for innlogging, noe som gjør systemet tilgjengelig for alle interesserte brukere.

## **Referanser**
- **MovieLens-datasettet**: Dataene i dette prosjektet ble hentet fra MovieLens-datasettet, nærmere bestemt MovieLens 20M (https://grouplens.org/datasets/movielens/20m/). Hentet 15.11.2024.
- **Gradio**: Brukt som hostingplattform (https://www.gradio.app/). Hentet 15.11.2024.
- **ChatGPT**: Brukt som inspirasjon og hjelpemiddel (https://chatgpt.com/). Hentet 15.11.2024.
- **freeCodeCamp**: Inspirasjon for prosjektet (https://www.freecodecamp.org/news/how-to-build-a-movie-recommendation-system-based-on-collaborative-filtering/). Hentet 15.11.2024.
