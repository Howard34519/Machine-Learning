## 概觀

本報告旨在詳細解釋人工智慧（AI）、機器學習（ML）和深度學習（DL）這三個密切相關但各有區別的技術領域。報告首先分別闡述每個概念的定義、核心原理、主要特徵、應用範例及優缺點，接著深入分析三者之間的層級關係與關鍵差異。總體而言，人工智慧是實現機器模擬人類智慧的廣泛目標；機器學習是達成此目標的一種重要方法，讓機器能從數據中學習；而深度學習則是機器學習的一個分支，利用深度神經網路處理更複雜的模式和任務 [1][11][18][45][49]。理解這三者的內涵及其關聯性，對於把握當前人工智慧發展的脈絡至關重要。

## 人工智慧（Artificial Intelligence, AI）

**定義與目標**
人工智慧（AI），亦稱「人工智能」或「機器智慧」，是指由人製造出來的機器所表現出來的智慧 [74][86]。其核心目標是建立能夠模擬、延伸甚至超越人類智慧能力的電腦系統或機器，使其能夠執行通常需要人類智慧才能完成的複雜任務，如感知、推理、學習、決策、解決問題、理解語言甚至發揮創造力 [69][71][72][75][78][87]。簡言之，AI 試圖讓電腦能夠像人類一樣思考和行動 [70][77][96]。

**發展與類型**
AI 的概念可追溯至 20 世紀中期，艾倫·圖靈（Alan Turing）提出了「機器能否思考」的疑問，而「人工智慧」一詞則在 1956 年的達特茅斯會議上正式誕生 [10][70][77][95][96]。此後，AI 的發展經歷了數次浪潮，研究重點從符號推理、專家系統，逐漸演變至現今以數據驅動的機器學習和深度學習為主 [10][78]。

目前的 AI 大致可分為兩類：
*   **狹隘人工智慧（Artificial Narrow Intelligence, ANI）或弱 AI**：這是目前唯一能實際應用的 AI 類型，專注於執行特定任務，例如語音助理（Siri、Alexa）、圖像識別、推薦系統、自動駕駛汽車的部分功能等 [10][70][75][78][81][95]。這些系統在特定領域可能表現出色，但不具備通用的人類認知能力或自我意識 [70][95]。
*   **通用人工智慧（Artificial General Intelligence, AGI）或強 AI**：指具備與人類同等智慧，能夠理解、學習和應用其智慧來解決任何問題的機器 [10][70][75][78][95]。AGI 目前仍處於理論和科幻階段，尚未實現 [69][70][75][81]。
*   **超級人工智慧（Artificial Superintelligence, ASI）**：指智慧水平遠遠超越最聰明人類的 AI [70]。這同樣是未來式的概念 [70]。

**應用領域**
AI 技術已滲透到各行各業及日常生活中 [70][77][96]，包括：
*   **自然語言處理（NLP）**：聊天機器人、語言翻譯、情感分析 [43][68][77][80]。
*   **電腦視覺（CV）**：圖像識別、物體偵測、人臉辨識、醫療影像分析 [3][9][42][70][77]。
*   **推薦系統**：影音串流平台、電子商務網站的個人化推薦 [11][43][45]。
*   **自動駕駛**：車輛導航、環境感知、決策控制 [3][9][43][75]。
*   **金融服務**：詐騙偵測、信用評分、演算法交易 [42][51][77]。
*   **醫療保健**：疾病診斷輔助、藥物研發、個人化健康管理 [41][42][77]。
*   **生成式 AI**：創作文本、圖像、音樂、程式碼等新內容 [68][70][80][85]。

**優缺點**
*   **優點**：能高效處理海量數據、自動化重複性任務、提供穩定一致的效能、輔助甚至做出更明智的決策、解決複雜問題 [70][73][95]。
*   **缺點**：研發和部署成本高昂、需要大量高品質數據、對專業人才需求高、可能存在演算法偏見、引發就業和倫理方面的擔憂 [70][95]。

## 機器學習（Machine Learning, ML）

**定義與核心概念**
機器學習是人工智慧的一個重要分支或實現方法 [1][5][11][16][18][42][44][45][46][48][49][52][53][57][61][63][66][72][95]。它專注於開發能夠讓電腦系統從數據中「學習」的演算法和統計模型，而無需進行明確的編程來指示如何執行特定任務 [11][15][19][41][43][44][47][50][51][52][55][56][58]。其核心思想是利用演算法來分析數據、識別模式、從經驗中學習，並基於學習到的知識做出預測或決策 [1][5][11][41][42][45][46][49][51][54][56]。

**運作方式**
機器學習系統通常包含三個主要部分 [47]：
1.  **決策過程（Decision Process）**：演算法基於輸入數據（有標籤或無標籤）進行預測或分類 [47]。
2.  **誤差函數（Error Function）**：評估模型的預測準確性，通常透過比較預測結果與已知答案（在監督式學習中）來計算誤差 [47]。
3.  **模型優化過程（Model Optimization Process）**：根據誤差函數的回饋，調整模型內部參數（如權重），以最小化預測誤差，這個過程會反覆進行，直到模型的準確性達到預設標準 [47]。

**主要類型**
機器學習演算法主要可分為以下幾類：
*   **監督式學習（Supervised Learning）**：使用帶有標籤的數據集進行訓練，模型學習輸入與已知輸出之間的映射關係，用於分類（如垃圾郵件偵測）或迴歸（如房價預測）任務 [15][43][45][46][51][67]。
*   **非監督式學習（Unsupervised Learning）**：使用未標籤的數據進行訓練，模型自行探索數據中的結構和模式，常用於分群（如客戶細分）或關聯規則挖掘（如購物籃分析） [15][16][43][45][46][51][67]。
*   **強化學習（Reinforcement Learning）**：模型（代理人）透過與環境互動，根據行動獲得的獎勵或懲罰來學習最佳策略，常用於遊戲（如 AlphaGo）或機器人控制 [4][16][43][46][51]。
*   **半監督式學習（Semi-supervised Learning）**：介於監督式和非監督式之間，利用少量標籤數據和大量未標籤數據進行學習 [2][46]。

**應用領域**
機器學習的應用十分廣泛 [46][51]，涵蓋：
*   **推薦引擎**：如 Netflix、Spotify [11][43][45]。
*   **自然語言處理**：情感分析、文本分類 [42][43][58]。
*   **電腦視覺**：人臉辨識、物體識別 [42][43][46]。
*   **預測分析**：天氣預報、股票市場分析、銷售預測 [46][49][51][58]。
*   **醫療診斷**：輔助判讀醫學影像、預測疾病風險 [41][46][56]。
*   **金融風控**：信用評分、反欺詐 [42][51][77]。

**優缺點**
*   **優點**：能從大量數據中發現隱藏模式和趨勢、自動化決策過程、模型能持續學習和改進、可處理多種數據格式 [45][51]。
*   **缺點**：需要大量且高品質的訓練數據、對於複雜任務可能需要人工特徵工程、模型的解釋性有時較差、容易受到數據偏見的影響 [5][16][21][58]。

## 深度學習（Deep Learning, DL）

**定義與核心概念**
深度學習是機器學習的一個特定分支或子領域，近年來取得了突破性進展 [1][2][3][4][5][6][11][14][15][16][17][18][21][22][23][24][25][27][29][30][32][33][34][35][38][39][40][44][46][49][81]。它的核心是使用一種稱為「人工神經網路（Artificial Neural Networks, ANNs）」的架構，特別是具有多個處理層（隱藏層）的深度神經網路（Deep Neural Networks, DNNs） [1][2][3][4][6][7][13][14][16][17][18][19][21][22][30][31][37][49]。這種多層結構模仿了人類大腦處理資訊的方式，允許模型從原始數據中逐層學習和提取日益複雜的特徵或表徵 [1][2][7][8][10][13][17][22][49]。

![ANN 結構示意圖 [95]](https://www.solomon-3d.com/wp-content/uploads/image-6.jpg)

**運作方式與架構**
深度學習模型通常由三種層組成 [1][7][31]：
*   **輸入層（Input Layer）**：接收原始數據，如圖像的像素值或文本的詞向量 [1][7][31]。
*   **隱藏層（Hidden Layers）**：位於輸入層和輸出層之間，負責對數據進行轉換和特徵提取。深度學習的「深度」正來源於其擁有多個（甚至數百個）隱藏層 [1][2][4][7][17][31]。每一層的輸出作為下一層的輸入，逐層進行更抽象的表示學習 [1][7][10]。
*   **輸出層（Output Layer）**：產生最終的預測結果，如圖像的分類標籤或翻譯後的句子 [1][7][31]。

資料在網路中從輸入層流向輸出層，經過各層節點（神經元）的加權計算和非線性轉換（透過激勵函數），這個過程稱為前向傳播 [3][7][10]。訓練過程中，模型會根據預測誤差，利用反向傳播演算法和梯度下降法來調整各層之間的連接權重，以提高準確性 [2][10]。

**與傳統機器學習的區別**
相較於傳統機器學習方法，深度學習的主要特點包括：
*   **自動特徵工程**：DL 模型能自動從原始數據中學習相關特徵，減少了對人工設計特徵的依賴 [4][5][9][16][19][21]。
*   **處理非結構化數據**：在處理圖像、語音、自然語言等非結構化數據方面表現出色 [3][5][9][15][16][19][21][23]。
*   **數據量需求**：通常需要非常大的數據集才能達到良好性能 [5][15][16][21][23][28]。
*   **計算資源需求**：訓練深度模型計算成本高昂，常需要 GPU 等高性能硬體支援 [2][16][21][23]。
*   **複雜性**：模型結構更複雜，有時難以解釋其內部決策過程（黑盒子問題） [16][21][43]。

**應用領域**
深度學習在許多領域取得了頂尖成果 [2][3][9][15][16]：
*   **電腦視覺**：圖像分類、物體偵測、圖像生成 [2][4][5][7][9][15][16]。
*   **語音識別**：語音轉文字、語音助手（如 Alexa） [2][4][9][15][16][19]。
*   **自然語言處理**：機器翻譯、文本生成、問答系統 [2][3][9][15][16][17][19]。
*   **自動駕駛汽車**：感知周圍環境、決策控制 [3][9][15]。
*   **醫學影像分析**：輔助診斷癌症等疾病 [5][9][16][21]。
*   **生成式 AI**：如 ChatGPT、圖像生成模型等 [1][9][85]。

**優缺點**
*   **優點**：在複雜任務（尤其是感知任務）上準確率高、能自動學習有效特徵、模型可擴展性強 [15][19][23]。
*   **缺點**：需要大量標註數據、計算資源消耗大、訓練時間長、模型可解釋性較差、對超參數敏感 [16][21][23]。

## AI、ML 與 DL 的關係

人工智慧（AI）、機器學習（ML）和深度學習（DL）三者之間存在清晰的層級和包含關係 [1][6][10][11][15][16][17][18][20][23][29][33][36][39][44][45][46][48][49][95]。

*   **AI 是最廣泛的概念和目標**：旨在創造能夠展現類人智慧的機器或系統 [1][10][16][45][49][71][95]。
*   **ML 是實現 AI 的一種途徑**：它是 AI 的一個子集，專注於讓機器透過從數據中學習來獲得智慧，而不需要明確的指令 [1][11][16][18][44][45][48][49][52][61][63][95]。
*   **DL 是 ML 的一個分支或特定技術**：它是 ML 的一個子集，利用具有多層結構的深度神經網路來實現學習，特別擅長處理複雜模式和非結構化數據 [1][2][11][14][16][17][18][21][22][23][29][33][44][46][49][81][95]。

可以將它們想像成一組同心圓：最外層是 AI，中間層是 ML，最內層是 DL [10][17][49]。或者可以理解為一種進化關係：ML 是 AI 的一種實現方式，而 DL 則是 ML 技術的一種進化或進階版本 [1][11][16][34][95]。

![AI、ML、DL 關係圖 [17]](https://www.researchgate.net/profile/Tianming-Liu-6/publication/338416182/figure/fig1/AS:844727235018752@1578410527541/Relations-among-artificial-intelligence-machine-learning-and-deep-learning.png)

下表總結了三者的主要區別：

| 特徵       | 人工智慧 (AI)                             | 機器學習 (ML)                               | 深度學習 (DL)                                         |
| :--------- | :---------------------------------------- | :------------------------------------------ | :---------------------------------------------------- |
| **範疇**   | 最廣泛的概念，模擬人類智慧的總稱 [16][49] | AI 的子集，透過數據學習的方法 [11][18][48]     | ML 的子集，使用深度神經網路的方法 [1][2][11][18][21]     |
| **目標**   | 讓機器具備思考和行動的能力 [1][77][96]     | 從數據中找出規律，進行預測或決策 [1][41][45] | 學習數據的多層次表示，處理複雜模式 [2][7][17][22][49] |
| **核心技術** | 包含 ML、DL、邏輯推理、規劃等多種技術 [78][81] | 各式學習演算法 (決策樹、SVM、迴歸等) [16][51][53] | 深度神經網路 (CNN, RNN, Transformer 等) [2][15][16][17] |
| **數據需求** | 不一定依賴大數據 (如早期專家系統)         | 需要數據進行訓練 [5][11][42]                    | 通常需要非常大量的數據 [5][16][21][23][28]              |
| **特徵工程** | 依賴具體技術                              | 常需人工設計或選擇特徵 [5][16][21][53]          | 可自動學習特徵 [4][9][16][19][21]                      |
| **計算需求** | 差異很大                                  | 相對較低 [21][53]                           | 非常高，常需 GPU [2][16][21][23]                      |
| **主要應用** | 涵蓋所有 AI 應用領域 [77]                | 預測、分類、推薦、模式識別 [11][41][45][51] | 圖像/語音識別、NLP、生成任務 [2][3][9][15][16][17][19] |
[1] https://solwen.ai/posts/deep-learning
[2] https://zh.wikipedia.org/zh-tw/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0
[3] https://www.ibm.com/think/topics/deep-learning
[4] https://www.ibm.com/think/topics/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks
[5] https://aws.amazon.com/tw/compare/the-difference-between-machine-learning-and-deep-learning/
[6] https://en.wikipedia.org/wiki/Deep_learning
[7] https://baubimedi.medium.com/%E9%80%9F%E8%A8%98ai%E8%AA%B2%E7%A8%8B-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E5%85%A5%E9%96%80-%E4%B8%80-68e27912ce30
[8] https://www.ecloudvalley.com/tw/blog/difference-between-ai-ml-dl
[9] https://aws.amazon.com/what-is/deep-learning/
[10] https://blogs.nvidia.com/blog/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/
[11] https://www.zendesk.tw/blog/machine-learning-and-deep-learning/
[12] https://www.datacamp.com/tutorial/machine-deep-learning
[13] https://ikala.ai/zh-tw/blog/uncategorized-zh/how-deep-learning-works/
[14] https://www.cnblogs.com/hlkawa/p/17629945.html
[15] https://cloud.google.com/discover/what-is-deep-learning
[16] https://digital.neweratech.com/articles/artificial-intelligence-machine-learning-and-deep-learning-overview
[17] https://substack.com/home/post/p-148330385?utm_campaign=post&utm_medium=web
[18] https://zhuanlan.zhihu.com/p/609397367
[19] https://www.geeksforgeeks.org/introduction-deep-learning/
[20] https://medium.com/@jereljohnvelarde/whats-the-relationship-of-ai-ml-dl-and-generative-ai-1f4c8295432a
[21] https://aws.amazon.com/cn/compare/the-difference-between-machine-learning-and-deep-learning/
[22] https://blog.csdn.net/qq_28791753/article/details/143819889
[23] https://www.trentonsystems.com/en-us/resource-hub/blog/deep-learning
[24] https://synoptek.com/insights/it-blogs/data-insights/ai-ml-dl-and-generative-ai-face-off-a-comparative-analysis/
[25] https://business.adobe.com/tw/products/real-time-customer-data-platform/deep-learning-vs-machine-learning.html
[26] https://aws.amazon.com/tw/what-is/deep-learning/
[27] https://link.springer.com/article/10.1007/s42979-021-00815-1
[28] https://www.turing.com/kb/ultimate-battle-between-deep-learning-and-machine-learning
[29] https://mlhowto.readthedocs.io/en/latest/deeplearn.html
[30] https://www.cake.me/resources/deep-learning
[31] https://www.ultralytics.com/glossary/deep-learning-dl
[32] https://www.quora.com/What-is-deep-learning-And-its-relation-between-machine-learning-and-artificial-intelligence
[33] https://www.cnblogs.com/BlogNetSpace/p/18229985
[34] https://www.solomon-3d.com/tw/blog/what-is-deep-learning/
[35] https://www.datacamp.com/tutorial/tutorial-deep-learning-tutorial
[36] https://www.cengn.ca/information-centre/innovation/difference-between-ai-ml-and-dl/
[37] https://www.oracle.com/tw/artificial-intelligence/machine-learning/what-is-deep-learning/
[38] https://www.zendesk.com/blog/machine-learning-and-deep-learning/
[39] https://blog.csdn.net/Xiebe/article/details/125416134
[40] https://www.techtarget.com/searchenterpriseai/definition/deep-learning-deep-neural-network
[41] https://aws.amazon.com/tw/what-is/machine-learning/
[42] https://www.netapp.com/zh-hant/artificial-intelligence/what-is-machine-learning/
[43] https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained
[44] https://aws.amazon.com/what-is/machine-learning/
[45] https://solwen.ai/posts/machine-learning
[46] https://medium.com/@troy801125/machine-learning-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%92%8C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%98%AF%E4%BB%80%E9%BA%BC-49a6ba41ab3e
[47] https://www.ibm.com/think/topics/machine-learning
[48] https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning
[49] https://www.sap.com/taiwan/products/artificial-intelligence/what-is-machine-learning.html
[50] https://developers.google.com/machine-learning/intro-to-ml/what-is-ml?hl=zh-tw
[51] https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-ml/
[52] https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-machine-learning
[53] https://www.intel.com.tw/content/www/tw/zh/learn/what-is-machine-learning.html
[54] https://zh.oosga.com/docs/machine-learning/
[55] https://enterprisersproject.com/article/2019/7/machine-learning-explained-plain-english
[56] https://zh.wikipedia.org/zh-cn/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0
[57] https://zh.wikipedia.org/zh-hant/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0
[58] https://en.wikipedia.org/wiki/Machine_learning
[59] https://www.zendesk.hk/blog/machine-learning-and-deep-learning/
[60] https://www.geeksforgeeks.org/ml-machine-learning/
[61] https://www.oracle.com/tw/artificial-intelligence/machine-learning/what-is-machine-learning/
[62] https://www.datacamp.com/blog/what-is-machine-learning
[63] https://www.opentext.com/zh-tw/what-is/machine-learning
[64] https://www.coursera.org/articles/what-is-machine-learning
[65] https://developers.google.com/machine-learning/guides/rules-of-ml?hl=zh-tw
[66] https://cloud.google.com/learn/what-is-machine-learning
[67] https://aws.amazon.com/tw/compare/the-difference-between-machine-learning-supervised-and-unsupervised/
[68] https://aws.amazon.com/tw/what-is/artificial-intelligence/
[69] https://www.heinz.cmu.edu/media/2023/July/artificial-intelligence-explained
[70] https://solwen.ai/posts/what-is-ai
[71] https://cloud.google.com/learn/what-is-artificial-intelligence?hl=zh-TW
[72] https://www.ibm.com/think/topics/artificial-intelligence
[73] https://www.kdan.com/zh-tw/blog/about/ai/
[74] https://zh.wikipedia.org/zh-tw/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD
[75] https://www.coursera.org/articles/what-is-artificial-intelligence
[76] https://ikala.cloud/blog/ai-learing/ai-introduction-trends
[77] https://www.netapp.com/zh-hant/artificial-intelligence/what-is-artificial-intelligence/
[78] https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-ai
[79] https://www.appier.com/zh-tw/blog/far-explainable-artificial-intelligence
[80] https://aws.amazon.com/cn/what-is/artificial-intelligence/
[81] https://cloud.google.com/learn/what-is-artificial-intelligence
[82] https://www.sap.com/taiwan/products/artificial-intelligence/what-is-artificial-intelligence.html
[83] https://www.oracle.com/tw/artificial-intelligence/what-is-ai/
[84] https://www.techtarget.com/searchenterpriseai/definition/AI-Artificial-Intelligence
[85] https://aws.amazon.com/tw/what-is/generative-ai/
[86] https://zh.wikipedia.org/zh-cn/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD
[87] https://www.nasa.gov/what-is-artificial-intelligence/
[88] https://www.cloudflare.com/zh-tw/learning/ai/what-is-artificial-intelligence/
[89] https://www.investopedia.com/terms/a/artificial-intelligence-ai.asp
[90] https://www.ibm.com/cn-zh/think/topics/artificial-intelligence
[91] https://www.britannica.com/technology/artificial-intelligence
[92] https://en.wikipedia.org/wiki/Artificial_intelligence
[93] https://www.opentext.com/zh-tw/what-is/artificial-intelligence
[94] https://www.sap.com/taiwan/products/artificial-intelligence/what-is-artificial-intelligence.html
[95] https://www.solomon-3d.com/tw/blog/what-is-ai/
[96] https://www.netapp.com/zh-hant/artificial-intelligence/what-is-artificial-intelligence/
[97] https://cloud.google.com/learn/what-is-artificial-intelligence?hl=zh-TW
[98] https://zh.wikipedia.org/zh-tw/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD
[99] https://ikala.cloud/blog/ai-learing/ai-introduction-trends
[100] https://ai4kids.ai/blogs/blog/about-artificial-intelligence-ai
[101] https://www.cloudflare.com/zh-tw/learning/ai/what-is-artificial-intelligence/
[102] https://aws.amazon.com/tw/what-is/artificial-intelligence/
[103] https://solwen.ai/posts/what-is-ai
[104] https://aws.amazon.com/cn/what-is/artificial-intelligence/
[105] https://www.ibm.com/cn-zh/think/topics/artificial-intelligence
[106] https://www.oracle.com/tw/artificial-intelligence/what-is-ai/
[107] https://www.techtarget.com/searchenterpriseai/definition/AI-Artificial-Intelligence
[108] https://www.heinz.cmu.edu/media/2023/July/artificial-intelligence-explained
[109] https://www.ibm.com/think/topics/artificial-intelligence
[110] https://www.coursera.org/articles/what-is-artificial-intelligence
[111] https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-ai
[112] https://cloud.google.com/learn/what-is-artificial-intelligence
[113] https://www.nasa.gov/what-is-artificial-intelligence/
[114] https://www.investopedia.com/terms/a/artificial-intelligence-ai.asp
[115] https://www.britannica.com/technology/artificial-intelligence
[116] https://en.wikipedia.org/wiki/Artificial_intelligence
[117] https://meng.uic.edu/news-stories/ai-artificial-intelligence-what-is-the-definition-of-ai-and-how-does-ai-work/
[118] https://www.intel.com.tw/content/www/tw/zh/learn/what-is-machine-learning.html
[119] https://www.sap.com/taiwan/products/artificial-intelligence/what-is-machine-learning.html
[120] https://www.netapp.com/zh-hant/artificial-intelligence/what-is-machine-learning/
[121] https://www.oracle.com/tw/artificial-intelligence/machine-learning/what-is-machine-learning/
[122] https://solwen.ai/posts/machine-learning
[123] https://medium.com/@troy801125/machine-learning-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%92%8C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%98%AF%E4%BB%80%E9%BA%BC-49a6ba41ab3e
[124] https://en.wikipedia.org/wiki/Machine_learning
[125] https://www.elastic.co/cn/what-is/machine-learning
[126] https://aws.amazon.com/tw/what-is/machine-learning/
[127] https://www.ibm.com/think/topics/machine-learning
[128] https://developers.google.com/machine-learning/intro-to-ml/what-is-ml?hl=zh-tw
[129] https://zh.wikipedia.org/zh-tw/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0
[130] https://zh.oosga.com/docs/machine-learning/
[131] https://www.bitdeer.ai/zh/blog/what-is-machine-learning-understanding-its-mechanisms-and-impact/
[132] https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained
[133] https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-ml/
[134] https://www.datacamp.com/blog/what-is-machine-learning
[135] https://kili-technology.com/data-labeling/machine-learning/machine-learning-defined-and-explained
[136] https://enterprisersproject.com/article/2019/7/machine-learning-explained-plain-english
[137] https://aws.amazon.com/what-is/machine-learning/
[138] https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning
[139] https://www.geeksforgeeks.org/ml-machine-learning/
[140] https://cloud.google.com/learn/what-is-machine-learning
[141] https://aws.amazon.com/tw/compare/the-difference-between-machine-learning-supervised-and-unsupervised/
[142] https://blogs.nvidia.cn/blog/what-is-a-machine-learning-model/
[143] https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-machine-learning
[144] https://zh.wikipedia.org/zh-tw/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0
[145] https://www.solomon-3d.com/tw/blog/what-is-deep-learning/
[146] https://masterconcept.ai/zh-hant/blog/ai-ml-dl-what-are-they-custom-enterprise-specific-generative-ai-models-using-vertex-ai/
[147] https://blog.csdn.net/qq_28791753/article/details/143819889
[148] https://www.ecloudvalley.com/tw/blog/difference-between-ai-ml-dl
[149] https://www.cnblogs.com/hlkawa/p/17629945.html
[150] https://solwen.ai/posts/deep-learning
[151] https://blog.csdn.net/Xiebe/article/details/125416134
[152] https://www.zendesk.com/blog/machine-learning-and-deep-learning/
[153] https://link.springer.com/article/10.1007/s42979-021-00815-1
[154] https://www.datacamp.com/tutorial/tutorial-deep-learning-tutorial
[155] https://ikala.ai/zh-tw/blog/uncategorized-zh/how-deep-learning-works/
[156] https://baubimedi.medium.com/%E9%80%9F%E8%A8%98ai%E8%AA%B2%E7%A8%8B-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E5%85%A5%E9%96%80-%E4%B8%80-68e27912ce30
[157] https://www.techtarget.com/searchenterpriseai/definition/deep-learning-deep-neural-network
[158] https://aws.amazon.com/tw/what-is/deep-learning/
[159] https://zhuanlan.zhihu.com/p/609397367
[160] https://www.ibm.com/think/topics/deep-learning
[161] https://www.oracle.com/tw/artificial-intelligence/machine-learning/what-is-deep-learning/
[162] https://en.wikipedia.org/wiki/Deep_learning
[163] https://aws.amazon.com/what-is/deep-learning/
[164] https://cloud.google.com/discover/what-is-deep-learning
[165] https://www.trentonsystems.com/en-us/resource-hub/blog/deep-learning
[166] https://www.geeksforgeeks.org/introduction-deep-learning/
[167] https://www.ultralytics.com/glossary/deep-learning-dl
[168] https://www.datacamp.com/tutorial/machine-deep-learning
[169] https://www.ecloudvalley.com/tw/blog/difference-between-ai-ml-dl
[170] https://blogs.nvidia.com/blog/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/
[171] https://www.quora.com/What-is-the-relationship-between-artificial-intelligence-machine-learning-deep-learning-and-artificial-neural-networks
[172] https://www.clicdata.com/blog/ai-ml-data-science-deep-learning/
[173] https://www.markreadfintech.com/p/d52
[174] https://ikala.cloud/blog/ai-learing/ml-1-ai-ml-deep-learning-intro
[175] https://u9534056.medium.com/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E5%AE%83%E5%80%91%E5%88%B0%E5%BA%95%E5%B7%AE%E5%9C%A8%E5%93%AA-de8ca30dca52
[176] https://www.sas.com/en_us/insights/articles/big-data/artificial-intelligence-machine-learning-deep-learning-and-beyond.html
[177] https://cloud.google.com/discover/deep-learning-vs-machine-learning?hl=zh-TW
[178] https://www.ibm.com/think/topics/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks
[179] https://cloud.google.com/discover/deep-learning-vs-machine-learning
[180] https://blog.mobagel.com/zh/difference-between-ai-ml-dl/
[181] https://www.coursera.org/articles/ai-vs-deep-learning-vs-machine-learning-beginners-guide
[182] https://www.sciencedirect.com/science/article/pii/S2667241323000113
[183] https://mile.cloud/zh/resources/blog/What-is-artificial-intelligence-machine-learning-deep%20learning_29
[184] https://solwen.ai/posts/deep-learning
[185] https://builtin.com/artificial-intelligence/ai-vs-machine-learning
[186] https://www.simplilearn.com/tutorials/artificial-intelligence-tutorial/ai-vs-machine-learning-vs-deep-learning
[187] http://www.360doc.com/content/21/0607/06/54396214_980800932.shtml
[188] https://www.tagtoo.com/blog/AI
