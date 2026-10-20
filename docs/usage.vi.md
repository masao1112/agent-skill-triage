# Sử dụng nhanh

Mở ứng dụng bằng `./run.sh`. Chọn file ZIP vừa tải trong Downloads, hoặc dùng nút
**Thư mục…** để kiểm tra toàn bộ gói đã giải nén có SKILL.md ở gốc.
Chọn riêng SKILL.md chỉ đọc file đó; chọn thư mục/ZIP sẽ đọc cả scripts/references.

1. **Xem nội dung**: xem danh sách file được đọc/bỏ qua và nội dung serialization.
   Bước này không cần model; không chạy bất kỳ script nào của gói.
2. **Analyze**: chạy classifier local; model được nạp một lần và tái sử dụng.
3. **Lưu kết quả JSON**: lưu vào `reports/` theo tên có thời gian; thư mục bị Git bỏ qua.
4. **Clear**: xóa nhãn/trạng thái cũ trước lần kiểm tra khác.

Có thể mở sẵn gói từ terminal:

```bash
./run.sh "$HOME/Downloads/skill.zip"
./run.sh "$HOME/Downloads/skill-folder" --analyze
```

Nút **Samples…** mở `samples/local/`; chọn một thư mục skill trong nhóm tương ứng.
Các sample nhập từ bộ nghiên cứu giữ nguyên bản gốc, có manifest/provenance và
SHA256 tại máy, được loại khỏi Git. Nhãn trong tên nhóm là nhãn nguồn, không phải
kết luận mới do người phát triển kiểm định. sample_meta.json không đưa vào model.

Fresh clone dùng hai file `.example.json`; model/base và adapter/tokenizer phải
được chuẩn bị riêng theo README. settings.json và inference_contract.json là cấu
hình cá nhân, không commit; prompt/cấu hình chung chia sẻ qua bản example.
Mã ứng dụng dùng MIT; quyền phân phối model và sample theo nguồn riêng.

HIGH_RISK nghĩa là cần rà soát trước khi cài, NORMAL không đảm bảo an toàn.
Prompt triển khai hiện tại chưa được xác nhận là prompt huấn luyện gốc. Kết quả
minh họa và unit test không thay thế đánh giá trên tập test độc lập.
