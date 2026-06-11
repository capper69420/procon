# procon

1. algorithm saijruulna(fps)
2. tatah, stickman igg huduguungu bailgah jishee
3. unah hurd tootsooloh
4. nuuriig tanij hucilturugc hemjdgee saijruulah

1. mashin surgah ai \muuguntsur bakter hugts\
2. zahialga gargah \ heregtei tehnik tuhuurumjiinhuu\
3. bakter mugntsr zergee haraad ilruuleh

Нийгмийн асуудал: Яагаад энэ төслийг хийх болсон шалтгаан (Work style reform, хөгшрөлт).

Технологийн шийдэл: Чиний өмнө нь хийж байсан YOLO, FaceMesh болон шинээр нэмэгдэх Whisper AI-ийн үүрэг.

Шүүгчдийг гайхшруулах гол цэг: "Task-Shifting" буюу ажил шилжүүлэлт хэрхэн явагдаж байгааг харуулсан хүснэгт.

Төлөвлөгөө: Багийнхан хэн, юуг, хэдий хугацаанд хийх тухай хөгжүүлэлтийн дараалал.

Дүгнэлт: Энэ төсөл яагаад Kosen Procon дээр ялах боломжтойг уриалсан санамж зэргийг бүрэн багтаасан болно.

To get a 100% free API with zero usage caps, zero subscription fees, and complete data privacy, the solution is to use an open-source model. You download the model weights for free, wrap them in a lightweight API server, and run it on your own computer or cloud server.For 70+ languages, the open-source ecosystem has two massive players that easily hit your requirement.1. OpenAI Whisper (Large-v3 / Turbo)Despite OpenAI being a commercial company, they fully open-sourced the weights for the Whisper model under the permissive MIT license.  Language Count: Supports 99+ languages out of the box.  Performance: It is the undisputed gold standard for open-source speech recognition. It effortlessly handles accents, automatically detects what language is being spoken, and can even translate those 99+ languages directly into English text.  Efficiency: Highly optimized variations like Whisper Large-v3 Turbo or Distil-Whisper give you top-tier accuracy while running incredibly fast on consumer hardware or budget-friendly GPUs.  2. Meta MMS (Massive Multilingual Speech)If Whisper doesn't cover a highly specific regional dialect you need, Meta's open-source MMS model is the ultimate backup.Language Count: Supports over 1,100 languages.  Performance: While Whisper is generally preferred for standard conversational accuracy in major world languages, Meta MMS is entirely open, free, and completely unmatched in its linguistic diversity.How to Turn These Models into Your Own Free APIYou don't have to build an API framework from scratch. The developer community has created drop-in API wrappers that make these open-source models behave exactly like a paid cloud service.Here are the best tools to deploy your own zero-cost API:faster-whisper-server (Highly Recommended): This open-source project wraps an optimized version of Whisper inside a web server. It creates a local API endpoint (http://localhost:8000/v1/audio/transcriptions) that mimics OpenAI's official API layout. You can switch your app from OpenAI's paid tier to your own free server by changing just one line of code.  Whisper.cpp (Server Mode): If you are running on a Mac (Apple Silicon) or standard CPU hardware without a massive dedicated graphics card, Whisper.cpp is a C/C++ port optimized for pure speed. It features a built-in server tool to spin up a local API endpoint immediately.  Hugging Face Transformers: Both Whisper and Meta MMS are hosted for free on Hugging Face. You can use Python with the transformers library and FastAPI to create a custom API endpoint in less than 20 lines of code.Summary  To get 100% free, unlimited speech-to-text for 70+ languages, download faster-whisper-server and run the Whisper Large-v3 Turbo model locally. Your only cost will be the electricity to power your machine.


Багийн 4 гишүүн үүрэг үүргээрээ (1-AI, 2-Data, 3-UI, 4-Document) яг юу хийхийг өдөр өдрөөр нь хуваасан тул үүнийг шууд ашиглаж болно.
🗓️ 1-Р ДОЛОО ХОНОГ: Суурь хөгжүүлэлт ба Эхний нэгтгэл (Alpha Version)
    Өдөр 1: Төлөвлөлт ба Орчин бэлтгэх (Kick-off & Setup)
1 (AI): MediaPipe FaceMesh болон YOLO-ийн орчныг бэлдэх, туршилтын жижиг скрипт ажиллуулж үзэх. 
2 (Data): Өгөгдлийн сангийн (Firebase эсвэл Supabase) бүтцийг үүсгэх. Өвчтөн болон Эмчийн хүснэгтүүдийг зохион байгуулах.
3 (UI): Figma дээр Киоск машины дэлгэц болон Эмчийн Дашбордын ерөнхий загварыг (Wireframe) зурах.
4 (Document): Прокон тэмцээний удирдамжтай дахин танилцаж, бүрдүүлэх бичиг баримт болон танилцуулгын бүтцийг гаргах.
    Өдөр 2-3: Үндсэн алгоритм ба Дизайн (Core Development)
1 (AI): Камераар өвдөлтийн зэрэг (Pain level) болон ил шарх/тэргэнцэр таних логикоо бичиж дуусгах.  
2 (Data): Хэрэглэгчийн мэдээллийг бааз руу хадгалах болон буцааж дуудах үндсэн API (Endpoints)-уудыг бичих.
3 (UI): Өндөр настнуудад зориулсан "Voice-First" (Том үсэгтэй, ойлгомжтой) Киоск дэлгэцийн кодыг (React/Vue г.м) эхлүүлэх.  
4 (Document): Системийн архитектурын диаграмыг зурах, эмнэлгийн "Task-Shifting" буюу ажил хөнгөвчилж буйг батлах онолын хэсгийг бичих. 
    Өдөр 4-5: Дуут AI болон Логик холболт (Voice Integration & Logic)
1 (AI): OpenAI Whisper API ашиглан яриаг текст рүү, ChatGPT ашиглан эмнэлгийн түүх (EHR) болгон хувиргах хэсгийг кодлох. 
2 (Data): AI-аас орж ирсэн датаг анализ хийж, өвчтөнийг Level A (Энгийн), Level B (Дунд), Level C (Яаралтай) руу автоматаар ангилах бэкэнд логикийг тохируулах.  
3 (UI): Эмчид зориулсан "Эрсдэлтэй өвчтөний мэдээлэл харах Дашборд" дэлгэцийг кодлох.  
4 (Document): Тэмцээнд танилцуулах үзүүлэнгийн (Pitch deck) слайдуудыг бэлтгэж эхлэх, ярих үгсийн ноорог (Script) гаргах.
    Өдөр 6-7: Эхний нэгтгэл (Alpha Integration)
1 (AI) & 2 (Data): AI моделиудын өгөгдлийг (Камерын дата + Дууны дата) бааз руу амжилттай, алдаагүй илгээдэг болгож холбох.
3 (UI): Баазаас мэдээллийг бодит цагаар (Real-time) татаж Дашборд дээр харуулах.
4 (Document): Эхний хувилбар дээр тулгуурлан техникийн бичиг баримтын "Хэрэгжүүлэлт" хэсгийг бичиж дуусгах.

🗓️ 2-Р ДОЛОО ХОНОГ: Сайжруулалт, Тест болон Тайзны бэлтгэл
    Өдөр 8-9: Нарийвчилсан сайжруулалт (Refinement & UX Polish)
1 (AI): Гэрэл муутай үед камер хэрхэн ажиллахыг шалгах, яриа таних AI-ийн хурд болон нарийвчлалыг сайжруулах.
2 (Data): Системийн ачааллыг багасгах, алдаа гарсан үед систем гацахгүй байх (Error handling) хамгаалалт хийх.
3 (UI): Дуу хоолойгоор ярихад дэлгэцэн дээр долгион гарах гэх мэт хэрэглэгчид ээлтэй анимаци (Visual feedback) нэмж өнгөлөх.
4 (Document): Прокон тэмцээний албан ёсны маягтуудыг бөглөж, төслийн тайланг эцсийн байдлаар хянах.
    Өдөр 10-11: Бүтэн Системийн Тест (End-to-End Testing - "Bug Hunt")
Бүх гишүүд хамтдаа: "Өвчтөн орж ирэх -> Камер таних -> Ярих -> Level C яаралтай гэж тогтоогдох -> Эмчийн дэлгэцэнд дохио очих" гэсэн бүтэн процессыг олон удаа тестлэх.  Олдсон алдаануудыг (Bugs) жагсааж 1, 2, 3-р гишүүд тэр даруй засах.
    Өдөр 12: Техник хангамжийн угсралт (Hardware & Demo Prep)1 (AI) & 2 (Data): Бичсэн кодоо Проконы үзэсгэлэнд зориулж 40 минутын дотор ширээн дээр угсарч харуулах боломжтой таблет болон вэб камер бүхий төхөөрөмж дээрээ (Local эсвэл Cloud) бүрэн суурилуулах.  3 (UI) & 4 (Document): Танилцуулгын слайдыг (PPT) эцэслэх, тайзны дизайнд тохируулах.
    Өдөр 13: Илтгэлийн сургуулилт (Pitch Rehearsal)Бүх гишүүд хамтдаа: Тайзан дээр гарч байгаагаар төсөөлөн сургуулилт хийх.Хэн нь илтгэлээ ярих, хэн нь тайзан дээр "Өвчтөн" болж жүжиглэн Киоск дээр ярьж үзүүлэх, хэн нь "Эмч" болж Дашбордыг үзүүлэхээ маш тодорхой хуваарилж, хугацаандаа багтаж байгаа эсэхийг хэмжих.
    Өдөр 14: Эцсийн хяналт ба Илгээх (Final Polish & Submit)
4 (Document): Бүх бичиг баримт, код, танилцуулга видео/материалыг шалгаж удирдамжийн дагуу тэмцээний хороо руу илгээх.Бүгд: Сайн амарч, тайзан дээрх үзүүлбэртээ бэлдэх!
