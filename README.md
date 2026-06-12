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


📅 1-Р ДОЛОО ХОНОГ: Суурь архитектур ба Альфа (Эхний) хувилбар
Өдөр 1-2: Техникийн бэлтгэл ба Дизайн (Foundation)
AI/ML Engineer: MediaPipe FaceMesh болон YOLO-ийн сангуудыг суулгаж, вэб камераас дүрсийг алдаагүй унших код бичих.

Backend & Data Engineer: Express.js (эсвэл FastAPI) дээр суурь серверээ босгох. Өвчтөний мэдээлэл болон "Level A, B, C" ангиллыг хадгалах өгөгдлийн сангийн хүснэгтүүдийг (Schema) Firebase/PostgreSQL дээр үүсгэх.

Frontend Developer: Киоск дэлгэцийн эхний нүүр, ярианы цонх, эмчийн Дашбордын үндсэн араг ясыг (Layout) React/Vue дээр кодлох.

Project Manager / QA: Төслийн даалгавруудыг Trello дээр үүсгэх. Эмнэлгийн анхан шатны бүртгэлийн асуумжийн логик дүрмийг (ямар шинж тэмдэг илэрвэл Level C болох вэ г.м) баримтжуулж багт өгөх.

Өдөр 3-4: Модель хөгжүүлэлт ба API холболт (Core Logic)
AI/ML Engineer: Нүүрний булчингийн хөдөлгөөнөөр өвдөлтийн зэргийг тодорхойлох математик логик (Pain Score) кодоо бичих. OpenAI Whisper API-ийг холбож дууг текст рүү хөрвүүлэх.

Backend & Data Engineer: Фронтэндээс өвчтөний датаг хүлээж авах, AI-аас ирсэн ярианы текстийг ChatGPT рүү илгээж эмнэлгийн түүх болгон цэгцлэх бэкэнд логикийг (Endpoints) бичих.

Frontend Developer: Дизайны дагуу Киоскны дэлгэцийг том үсэгтэй, өндөр настнуудад ойлгомжтой өнгө төрхтэй болгож өөрчлөх. Товчлууруудыг Backend-тэй холбож эхлэх.

Project Manager / QA: Эхний бичиг баримтын "Оршил", "Нийгмийн асуудал" хэсгийг Проконы форматын дагуу бэлдэх. Хөгжүүлэгчдийн дунд өдөр бүр 10 минутын уулзалт хийж явцыг хянах.

Өдөр 5-7: Эхний нэгтгэл ба Туршилт (Alpha Integration)
AI/ML Engineer: Камерын визуал оноо болон Whisper-ийн текстийг нэгтгэж, Backend рүү илгээх бэлэн модуль (Payload) болгох.

Backend & Data Engineer: Ирсэн өгөгдөл дээр тулгуурлан өвчтөнийг Level A, B, эсвэл C рүү хуваарилдаг үндсэн алгоритмаа бичиж дуусгах.

Frontend Developer: Эмчийн Дашбордыг бэкэндтэй холбож, шинэ өвчтөн бүртгэгдмэгц дэлгэцэн дээр бодит цагаар (Real-time) мэдээлэл нь унадаг болгох.

Project Manager / QA: Анхны Альфа Туршилт. Камерны өмнө сууж яриад, эмчийн дэлгэцэнд мэдээлэл зөв очиж байгаа эсэхийг бүтэн шалгаж, олдсон алдаануудыг (Bugs) бүртгэх.

📅 2-Р ДОЛОО ХОНОГ: Сайжруулалт, Нарийн тест ба Тайзны бэлтгэл
Өдөр 8-9: Алдаа засах ба Өнгөлгөө (Beta Optimization)
AI/ML Engineer: Хүн хурдан эсвэл шивнээд ярихад Whisper хэрхэн таньж байгааг турших, өвдөлт илрүүлэх камерын нарийвчлалыг сайжруулах.

Backend & Data Engineer: Олон хүн зэрэг бүртгүүлэх үеийн серверийн ачааллыг тооцох, алдаа гарсан үед систем унахгүй байх хамгаалалт (Error Handling) хийх.

Frontend Developer: Хэрэглэгчийн интерфейсийг илүү "амьд" болгох. Дуу хоолойгоор ярьж байх үед дэлгэцэн дээр долгион үүсэх (Audio Visualizer) зэрэг UX анимациудыг нэмэх.

Project Manager / QA: Илтгэлийн слайдыг (Pitch Deck) бэлдэж эхлэх. Техникийн бичиг баримтын "Архитектур" болон "Хэрэглэсэн технологи" хэсгийг бичих.

Өдөр 10-11: Системийг Клоуд руу шилжүүлэх ба Стресс тест (Deployment)
AI/ML Engineer & Backend Engineer: Серверийг Клоуд (жишээ нь Heroku, Render эсвэл AWS) дээр байрлуулж, Киоск болон Эмчийн дашборд дэлхий хаанаас ч холбогдож ажиллах боломжтой болгох.

Frontend Developer: Системийн вэб хувилбарыг таблет эсвэл зөөврийн компьютер дээр ажиллуулж, дэлгэцийн хэмжээ алдагдаж байгаа эсэхийг (Responsive design) шалгах.

Project Manager / QA: Бүтэн Бета Туршилт. Багийн бүх гишүүд өөр өөр дүрд тоглож (Гадаад өвчтөн, Яаралтай өвчтөн, Эмч) системийг дор хаяж 20 удаа туршиж, ямар ч алдаагүй ажиллах хүртэл нь хөгжүүлэгчидтэй хамтран засах.

Өдөр 12-13: Үзүүлэн бэлдэх ба Илтгэлийн сургуулилт (Demo & Pitch)
AI/ML & Backend & Frontend: Проконы үзэсгэлэнгийн талбайд 40 минутын дотор угсарч харуулах компакт төхөөрөмжөө (Ноутбук + Вэб Камер + Микрофон) бэлдэж, сүлжээгүй эсвэл интернэт муу үед ажиллах локал хувилбарыг (Fallback) бэлэн байлгах.


Project Manager / QA: Илтгэлийг секундаар хэмжиж сургуулилт хийх. Тайзан дээр системээ хэрхэн "Амьдаар" (Live Demo) үзүүлэх жүжигчилсэн дарааллыг багийнхандаа зааж өгөх. Бичиг баримтыг эцсийн байдлаар хянах.

Өдөр 14: Эцсийн хяналт ба Илгээх (Submission)
Бүх бичиг баримт, код (GitHub линк), танилцуулга видеог шалгаж Проконы зохион байгуулах хороо руу албан ёсоор илгээх. Багаараа баяраа тэмдэглэх!







1. Huruuni hee ashiglan medelel awna
2. Speech to text
3. Ai model ypon helteig songoh
4. Delgets deer tovch gargaj ireh udirlagatai
