use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl BBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Self {
        BBox { x1, y1, x2, y2, score }
    }

    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    pub fn intersection(&self, other: &BBox) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);
        let width = (x2 - x1).max(0.0);
        let height = (y2 - y1).max(0.0);
        width * height
    }

    pub fn iou(&self, other: &BBox) -> f32 {
        let inter = self.intersection(other);
        let union = self.area() + other.area() - inter;
        inter / union
    }
}

pub fn nms(mut bboxes: Vec<BBox>, threshold: f32) -> Vec<BBox> {
    bboxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; bboxes.len()];

    for i in 0..bboxes.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(bboxes[i].clone());

        for j in (i + 1)..bboxes.len() {
            if suppressed[j] {
                continue;
            }

            if  bboxes[i].iou(&bboxes[j]) > threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}
