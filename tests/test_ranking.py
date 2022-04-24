from genz_tokenize.ranking import BM25, BM25Plus
from genz_tokenize.preprocess import vncore_tokenize, remove_punctuations
from vncorenlp import VnCoreNLP

vncore = VnCoreNLP(port=9000)

documents = [
    vncore_tokenize(remove_punctuations(
        'Năm 2013 , Nguyễn Quang Hải giành chức vô địch U21 quốc gia 2013 cùng với đội trẻ Hà Nội T&T và tạo nên cú sốc khi trở thành cầu thủ 16 tuổi đầu tiên giành được danh hiệu vô địch U21 quốc gia .'), vncore),
    vncore_tokenize(remove_punctuations(
        'Anh bắt đầu gia nhập lò đào tạo trẻ Hà Nội T&T khi mới 9 tuổi vào năm 2006 .'), vncore),
    vncore_tokenize(remove_punctuations('Cũng trong thập niên 1850 , các đội bóng nghiệp dư bắt đầu được thành lập và thường mỗi đội xây dựng cho riêng họ những luật chơi mới của môn bóng đá , trong đó đáng chú ý có câu lạc bộ Sheffield F.C .. Việc mỗi đội bóng có luật chơi khác nhau khiến việc điều hành mỗi trận đấu giữa họ diễn ra rất khó khăn .'), vncore),
    vncore_tokenize(remove_punctuations(
        'Quân đội Hoa Kỳ hay Các lực lượng vũ trang Hoa Kỳ là tổng hợp các lực lượng quân sự thống nhất của Hoa Kỳ . Các lực lượng này gồm có Lục quân , Hải quân , Thuỷ quân lục chiến , Không quân và Tuần duyên .'), vncore)
]

query = vncore_tokenize(remove_punctuations(
    'Quang Hải giành được chức vô địch U21 quốc gia năm bao nhiêu tuổi'), vncore)

bm25 = BM25(documents=documents)
print(bm25.get_score(query))
